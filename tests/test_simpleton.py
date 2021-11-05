import pytest
from simpleton.process import chunks


def test_chinks():
   rslt = chunks(lst= [1,2,3,4,5,6],n = 3)
   assert len(rslt) == 2