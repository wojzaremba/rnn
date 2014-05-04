#!/usr/bin/python
import sys
import unittest
from tests.mock_test import MockTests

def RunTests(with_buffer, test_number=-1, test_name=None):
  testmodules = [
    MockTests,
    ]
  if test_number != -1:
    testmodules = [testmodules[test_number]]
  totalTestsRun = 0
  for t in testmodules:
    suite = unittest.TestSuite()
    print "Running %s" % t.__name__
    if test_name is not None:
      suite.addTest(t(test_name))
    else:
      for method in dir(t):
        if method.startswith("test"):
          suite.addTest(t(method))
    result = unittest.TextTestRunner(verbosity=1, buffer=with_buffer).run(suite)
    totalTestsRun += result.testsRun
    if result.wasSuccessful():
      print "Tests %s were successful!" % t.__name__
    else:
      print "Tests %s failed" % t.__name__
      return False
  print "Total number of executed tests : ", totalTestsRun
  return True

def main():
  test_number = -1
  test_name = None
  if len(sys.argv) > 1:
    try:
      test_number = int(sys.argv[1])
    except ValueError:
      print "Input is not an int!"

  if len(sys.argv) > 2:
    test_name = sys.argv[2]
  RunTests(with_buffer=False, test_number=test_number, test_name=test_name)

if __name__ == '__main__':
  main()
