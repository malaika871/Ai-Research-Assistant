import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import config without allowing it to read the developer's local .env file.
with patch("dotenv.load_dotenv"):
    config = importlib.import_module("config")
    services = importlib.import_module("src.services")
    generator_module = importlib.import_module("src.generator")


class LLMBackendTest(unittest.TestCase):
    def setUp(self):
        services.ServiceHub._client = None
        services.ServiceHub._openai_client = None

    def tearDown(self):
        services.ServiceHub._client = None
        services.ServiceHub._openai_client = None

    def test_hf_backend_remains_shared_with_embeddings(self):
        expected_client = object()
        with (
            patch.object(config, "LLM_BACKEND", "hf"),
            patch.object(services, "InferenceClient", return_value=expected_client),
        ):
            llm_client = services.ServiceHub.get_llm_client()
            embedding_client = services.ServiceHub.get_embedding_client()

        self.assertIs(llm_client, expected_client)
        self.assertIs(embedding_client, expected_client)

    def test_openai_compat_client_uses_config_and_embeddings_stay_hf(self):
        created = {}

        def fake_openai(**kwargs):
            created.update(kwargs)
            return object()

        openai_module = ModuleType("openai")
        openai_module.OpenAI = fake_openai
        with (
            patch.object(config, "LLM_BACKEND", "openai_compat"),
            patch.object(config, "LLM_BASE_URL", "https://llm.example.invalid/v1"),
            patch.object(config, "LLM_API_KEY", "test-key"),
            patch.object(services, "InferenceClient", return_value="hf-client"),
            patch.dict(sys.modules, {"openai": openai_module}),
        ):
            llm_client = services.ServiceHub.get_llm_client()
            embedding_client = services.ServiceHub.get_embedding_client()

        self.assertEqual(
            created,
            {"base_url": "https://llm.example.invalid/v1", "api_key": "test-key"},
        )
        self.assertIsNot(llm_client, embedding_client)
        self.assertEqual(embedding_client, "hf-client")

    def test_generator_supports_standard_streaming_and_intent_calls(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="answer"),
                delta=SimpleNamespace(content="stream token"),
            )]
        )
        classification = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="GENERAL"))]
        )
        client = MagicMock()

        def create_completion(**kwargs):
            if kwargs.get("stream"):
                return iter([completion])
            if kwargs["messages"][0]["content"].startswith("Classify"):
                return classification
            return completion

        client.chat.completions.create.side_effect = create_completion
        with patch.object(services.ServiceHub, "get_llm_client", return_value=client):
            generator = generator_module.Generator()

        self.assertEqual(generator.generate("question", [])["answer"], "answer")
        self.assertEqual(list(generator.generate_stream("question", [])), ["stream token"])
        self.assertEqual(
            generator.generate_from_web("question", [])["answer"], "answer"
        )
        self.assertEqual(
            list(generator.generate_stream_from_web("question", [])), ["stream token"]
        )
        self.assertEqual(
            generator.generate_combined("question", [], [])["answer"], "answer"
        )
        self.assertEqual(
            list(generator.generate_stream_combined("question", [], [])),
            ["stream token"],
        )

        self.assertEqual(generator.classify_intent("question"), "GENERAL")

    def test_error_messages_include_backend_and_handle_rate_limits(self):
        generator = generator_module.Generator.__new__(generator_module.Generator)

        class RateLimitError(Exception):
            status_code = 429

        with (
            patch.object(config, "LLM_BACKEND", "openai_compat"),
            patch.object(config, "LLM_MODEL", "test-model"),
        ):
            rate_error = generator._format_generation_error(RateLimitError())
            model_error = generator._format_generation_error(
                RuntimeError("model not found")
            )

        self.assertIn("rate limit", rate_error.lower())
        self.assertIn("try again", rate_error.lower())
        self.assertIn("test-model", model_error)
        self.assertIn("openai_compat", model_error)

    def test_stream_iteration_rate_limit_has_retry_message(self):
        generator = generator_module.Generator.__new__(generator_module.Generator)

        class RateLimitError(Exception):
            status_code = 429

        client = MagicMock()

        def create_stream(**kwargs):
            def stream():
                raise RateLimitError("too many requests")
                yield "unreachable"

            return stream()

        client.chat.completions.create.side_effect = create_stream
        generator.client = client

        tokens = list(generator.generate_stream("question", []))
        self.assertEqual(len(tokens), 1)
        self.assertIn("rate limit", tokens[0].lower())
        self.assertIn("try again", tokens[0].lower())


if __name__ == "__main__":
    unittest.main()
