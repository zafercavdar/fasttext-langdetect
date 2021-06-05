from unittest import TestCase

from ftlangdetect import detect


class TestDetect(TestCase):

    def test_detect_low_mem(self):
        result = detect("Bugün hava çok güzel", low_memory=True)
        assert "lang" in result
        assert "score" in result
        assert isinstance(result["lang"], str)
        assert isinstance(result["score"], float)

    def test_detect_high_mem(self):
        result = detect("Bugün hava çok güzel", low_memory=False)
        assert "lang" in result
        assert "score" in result
        assert isinstance(result["lang"], str)
        assert isinstance(result["score"], float)
