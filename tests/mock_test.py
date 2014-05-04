import unittest
from main import mock

class MockTests(unittest.TestCase):

  def test_ambiguity0(self):
    self.run_mock(0)

  def test_ambiguity4(self):
    self.run_mock(4)

  def run_mock(self, backroll):
    model = mock(backroll)
    model.init(0, False)
    model.train()
    perplexity = model.test()
    self.assertLess(perplexity, 1.01)
    model.source.batch_size = 1
    model.start_it = model.load(float('inf'), False)
    threshold = 5e-3
    texts = list(set(model.gen("aa", threshold)))
    self.assertEqual(len(texts), 1)
    texts = list(set(model.gen("da", threshold)))
    self.assertEqual(len(texts), 1)
    texts = list(set(model.gen("a", threshold)))
    self.assertEqual(len(texts), 2)
    texts = list(set(model.gen("c", threshold)))
    self.assertEqual(len(texts), 2)

def main():
  unittest.main()

if __name__ == '__main__':
  main()
