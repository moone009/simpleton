import pytest
import pandas as pd
from simpleton.process import chunks
from simpleton.nlp import count_tokens

def make_dataframe():
    docs = [
            'WORK-FLEX HIGH DEXTERITY GLOVE',
            'RELAXED FIT WASHED DUCK SHERPA-LINED UTILITY JACKET',
            'LOOSE FIT WASHED DUCK INSULATED ACTIVE JAC',
            'SUPER DUX™ RELAXED FIT SHERPA-LINED ACTIVE JAC'
    ]
    df = pd.DataFrame({"ID":range(0,len(docs),1),"Docs":docs})
    return df

def test_chunks():
   rslt = chunks(lst= [1,2,3,4,5,6],n = 3)
   assert len(rslt) == 2

def test_count_tokens():
   df = make_dataframe()
   rslt = count_tokens(df['Docs'],3)
   assert len(rslt) == 19