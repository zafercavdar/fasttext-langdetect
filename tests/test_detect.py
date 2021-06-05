from unittest import TestCase

from ftlangdetect import detect


class TestDetect(TestCase):

    def test_detect_base(self):
        result = detect("Bugün hava çok güzel")
        assert "lang" in result
        assert "score" in result
        assert isinstance(result["lang"], str)
        assert isinstance(result["score"], float)
