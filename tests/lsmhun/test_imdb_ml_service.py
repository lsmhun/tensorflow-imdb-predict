from lsmhun.imdb_ml_service import ImdbMlService

sample_pred_text = 'The movie was cool. The animation and the graphics were out of this world. I would recommend this movie.'
sample_pred_text2 ='This film is shit. Bad actors, terrible story, visually nothing. I lost two hours from my life.'

def test_tokenization():
   global sample_pred_text
   victim = ImdbMlService()
   expected_result = [19, 27, 18, 2724, 3, 19, 1847, 5, 1, 5172, 8, 85, 61, 6, 14, 562, 3, 12, 70, 505, 14, 65, 7975]
   assert victim.tokenizer.encode(sample_pred_text) == expected_result

def test_positive_message():
   global sample_pred_text
   victim = ImdbMlService()
   result = victim.evaluate_text(sample_pred_text)
   # [[0.913821]]
   assert result[0][0] > 0.85

def test_positive_message():
   global sample_pred_text2
   victim = ImdbMlService()
   result = victim.evaluate_text(sample_pred_text2)
   # [[0.009932]]
   assert result[0][0] < 0.15
