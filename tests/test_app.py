import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app
from agno.models.google import Gemini


class ShoppingAppTests(unittest.TestCase):
    def test_extracts_budget_from_query_even_when_budget_control_is_any(self):
        self.assertEqual(
            app.extract_budget("A phone under ₹25,000 for my mother", "Any"),
            "Under ₹25,000",
        )

    def test_extracts_k_and_lakh_budget_shorthand(self):
        self.assertEqual(app.extract_budget("Laptop under 60k", "Any"), "Under ₹60,000")
        self.assertEqual(app.extract_budget("Laptop budget 1.2 lakh", "Any"), "Under ₹120,000")

    def test_does_not_invent_an_unspecified_budget(self):
        self.assertEqual(app.extract_budget("A laptop for college", "Any"), "Not specified")

    def test_parses_json_response(self):
        self.assertEqual(app.parse_agent_response('{"products": []}'), {"products": []})

    def test_rejects_invalid_json_response(self):
        with self.assertRaisesRegex(ValueError, "could not be parsed as JSON"):
            app.parse_agent_response("not JSON")

    def test_rejects_non_object_json_response(self):
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            app.parse_agent_response('["products"]')

    def test_normalizes_recommendation_aliases_and_product_fields(self):
        result = app.normalize_research_result(
            {
                "recommendations": [
                    {
                        "name": "Example laptop",
                        "key_strengths": ["Portable"],
                        "trade_off": "Limited ports",
                    }
                ],
                "brief": {"primary_use": "College"},
            }
        )

        self.assertEqual(result["brief"]["Primary use"], "College")
        self.assertEqual(result["products"][0]["strengths"], ["Portable"])
        self.assertEqual(result["products"][0]["tradeoff"], "Limited ports")

    def test_reads_provider_error_code_from_json_string(self):
        self.assertEqual(
            app.provider_error(
                '{"error":{"code":429,"message":"Quota exceeded"}}',
                "ERROR",
            ),
            ("Quota exceeded", 429),
        )

    def test_adds_only_provider_grounding_urls_to_sources(self):
        result = {"sources": [{"url": "https://example.com/a", "title": "A"}]}
        citations = SimpleNamespace(
            urls=[
                SimpleNamespace(url="https://example.com/a", title="A"),
                SimpleNamespace(url="https://example.com/b", title="B"),
            ]
        )

        app.add_grounding_sources(result, citations)

        self.assertEqual(
            result["sources"],
            [
                {"url": "https://example.com/a", "title": "A"},
                {"url": "https://example.com/b", "title": "B"},
            ],
        )

    def test_reports_error_returned_by_agno_instead_of_misreporting_missing_products(self):
        brief = {"Category": "Laptop", "Budget": "Not specified"}
        with patch.dict(os.environ, {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}):
            with patch.object(app, "Agent") as agent_class:
                with patch.object(app, "DuckDuckGoTools", return_value=object()):
                    agent_class.return_value.run.return_value.content = {
                        "error": "Requested model is unavailable"
                    }
                    with self.assertRaisesRegex(
                        RuntimeError, "Requested model is unavailable"
                    ):
                        app.do_live_research("Laptop for college", brief)

    def test_retries_with_fallback_model_after_gemini_503(self):
        response = app.ShoppingResearch.model_validate(
            {
                "brief": {"category": "Laptop"},
                "products": [{"name": "Example laptop"}],
                "decision_matrix": {},
                "trade_off": "Information unavailable.",
                "sources": [],
            }
        )
        failed_agent = Mock()
        failed_agent.run.return_value.content = {
            "error": {
                "code": 503,
                "status": "UNAVAILABLE",
                "message": "Temporarily unavailable",
            }
        }
        successful_agent = Mock()
        successful_agent.run.return_value.content = response
        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "gemini",
                "GEMINI_API_KEY": "test-key",
                "GEMINI_MODEL": "gemini-primary",
                "GEMINI_FALLBACK_MODEL": "gemini-fallback",
            },
        ):
            with patch.object(
                app, "Gemini", side_effect=lambda id, api_key, **kwargs: SimpleNamespace(id=id)
            ) as model_factory:
                with patch.object(
                    app, "Agent", side_effect=[failed_agent, successful_agent]
                ) as agent_factory:
                    with patch.object(app, "DuckDuckGoTools", return_value=object()):
                        result = app.do_live_research(
                            "Laptop for college", {"Budget": "Not specified"}
                        )

        self.assertEqual(len(result["products"]), 1)
        self.assertEqual(result["research_model"], "gemini-fallback")
        self.assertEqual(
            [call.kwargs["id"] for call in model_factory.call_args_list],
            ["gemini-primary", "gemini-fallback"],
        )
        self.assertEqual(agent_factory.call_count, 2)

    def test_explains_gemini_quota_error_when_fallbacks_are_exhausted(self):
        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "gemini",
                "GEMINI_API_KEY": "test-key",
                "GEMINI_MODEL": "gemini-primary",
                "GEMINI_FALLBACK_MODEL": "gemini-fallback",
            },
        ):
            with patch.object(
                app, "Gemini", side_effect=lambda id, api_key, **kwargs: SimpleNamespace(id=id)
            ):
                with patch.object(app, "Agent") as agent_class:
                    agent_class.return_value.run.return_value.content = (
                        '{"error":{"code":429,"message":"Quota exceeded"}}'
                    )
                    with patch.object(app, "DuckDuckGoTools", return_value=object()):
                        with self.assertRaisesRegex(RuntimeError, "Gemini API quota/rate limit"):
                            app.do_live_research(
                                "Laptop for college", {"Budget": "Not specified"}
                            )
        self.assertEqual(agent_class.call_count, 2)

    def test_missing_gemini_key_is_reported_instead_of_falling_back(self):
        with patch.dict(os.environ, {"AI_PROVIDER": "gemini"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY is missing"):
                app.do_live_research("Laptop for college", {"Budget": "Not specified"})

    def test_uses_only_gemini_38_when_no_fallbacks_are_configured(self):
        response = app.ShoppingResearch.model_validate(
            {
                "brief": {"category": "Laptop"},
                "products": [{"name": "Example laptop"}],
            }
        )
        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "gemini",
                "GEMINI_API_KEY": "test-key",
                "GEMINI_MODEL": "gemini-3.8-flash",
                "GEMINI_FALLBACK_MODEL": "",
            },
        ):
            with patch.object(
                app, "Gemini", side_effect=lambda id, api_key, **kwargs: SimpleNamespace(id=id)
            ) as model_factory:
                with patch.object(app, "Agent") as agent_class:
                    agent_class.return_value.run.return_value.content = response
                    result = app.do_live_research(
                        "Laptop for college", {"Budget": "Not specified"}
                    )

        self.assertEqual(result["products"][0]["name"], "Example laptop")
        self.assertEqual(
            [call.kwargs["id"] for call in model_factory.call_args_list],
            ["gemini-3.8-flash"],
        )

    def test_gemini_provider_passes_live_result_through_agno(self):
        response = {
            "brief": {"Category": "Laptop"},
            "products": [{"name": "Example laptop", "price": "Not verified"}],
            "decision_matrix": {"Criteria": ["Battery"], "Example laptop": ["Not verified"]},
            "trade_off": "Information unavailable.",
            "sources": [],
        }
        brief = {"Category": "Laptop", "Budget": "Not specified"}
        structured_response = app.ShoppingResearch.model_validate(response)
        with patch.dict(os.environ, {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}):
            with patch.object(app, "Agent") as agent_class:
                with patch.object(app, "DuckDuckGoTools", return_value=object()):
                    agent_class.return_value.run.return_value.content = structured_response
                    result = app.do_live_research("Laptop for college", brief)

        self.assertEqual(result["products"][0]["name"], "Example laptop")
        self.assertIsInstance(agent_class.call_args.kwargs["model"], Gemini)
        self.assertTrue(agent_class.call_args.kwargs["model"].search)
        self.assertIs(agent_class.call_args.kwargs["output_schema"], app.ShoppingResearch)
        self.assertTrue(agent_class.call_args.kwargs["structured_outputs"])


if __name__ == "__main__":
    unittest.main()
