import importlib.util
import os
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("twitter_airline_sentiment.py")


def load_application_module():
    loaded_paths = []
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.cache_resource = lambda function: function
    specification = importlib.util.spec_from_file_location(
        "twitter_airline_sentiment_under_test",
        MODULE_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the application module.")

    module = importlib.util.module_from_spec(specification)

    def record_model_path(artifact_file):
        loaded_paths.append(Path(artifact_file.name))
        return object()

    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory() as temporary_directory:
        os.chdir(temporary_directory)
        try:
            with (
                mock.patch.dict(sys.modules, {"streamlit": fake_streamlit}),
                mock.patch.object(pickle, "load", side_effect=record_model_path),
            ):
                specification.loader.exec_module(module)
        finally:
            os.chdir(original_directory)

    return module, loaded_paths


class TwitterAirlineSentimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application, cls.loaded_paths = load_application_module()

    def test_models_load_relative_to_the_application(self):
        expected_names = {
            "airline_vectorizer.pkl",
            "twitter_sentiment_pred.sav",
            "spam_mail_predict.sav",
            "spam_vectorizer.pkl",
        }

        self.assertEqual(len(self.loaded_paths), 4)
        self.assertEqual({path.name for path in self.loaded_paths}, expected_names)
        self.assertTrue(
            all(path.parent == MODULE_PATH.parent for path in self.loaded_paths)
        )

    def test_parses_csv_and_nested_json_exports(self):
        csv_tweets = self.application.parse_xquik_export(
            "tweets.csv",
            b"id,text\n1,Clear skies today\n2,Flight delayed\n",
        )
        json_tweets = self.application.parse_xquik_export(
            "tweets.json",
            b'{"data":[{"tweet":{"full_text":"Nested tweet"}}]}',
        )

        self.assertEqual(csv_tweets, ["Clear skies today", "Flight delayed"])
        self.assertEqual(json_tweets, ["Nested tweet"])

    def test_parses_xquik_production_csv_and_jsonl_exports(self):
        csv_tweets = self.application.parse_xquik_export(
            "tweets.csv",
            (
                b"Tweet Text,Username,Views,Likes,Replies,Tweet Created At\n"
                b"Production header,xquik,10,2,1,2026-07-19T00:00:00Z\n"
            ),
        )
        jsonl_tweets = self.application.parse_xquik_export(
            "tweets.jsonl",
            b'{"tweetText":"Camel case"}\n{"content":"Second post"}\n',
        )

        self.assertEqual(csv_tweets, ["Production header"])
        self.assertEqual(jsonl_tweets, ["Camel case", "Second post"])

    def test_rejects_an_empty_export(self):
        with self.assertRaisesRegex(ValueError, "Xquik export is empty"):
            self.application.parse_xquik_export("tweets.json", b"  ")


if __name__ == "__main__":
    unittest.main()
