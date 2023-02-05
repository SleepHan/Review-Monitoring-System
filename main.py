from aspect_abstraction import get_aspects
from ExtractReviewShopee import getReviews
from FakeReview import isol_preprocess, isol_trainer, isol_predict
from sgnlp.models.sentic_gcn import (
    SenticGCNBertTokenizer,
    SenticGCNBertEmbeddingConfig,
    SenticGCNBertEmbeddingModel,
    SenticGCNBertModel,
    SenticGCNBertPreprocessor,
    SenticGCNBertConfig,
    SenticGCNBertPostprocessor,
)
import pandas as pd
import numpy as np
import argparse
import re

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--csv', type=str, required=True)
# Parse the argument
args = parser.parse_args()

# Get shopee Review
# url = "https://shopee.sg/DERE-Laptop-T30-Tablet-Laptop-2-in-1-16GB-RAM-1TB-SSD-13-Inch-2K-IPS-Screen-Windows11-11th-Gen-Up-To-2.9GHz-Touch-Screen-Laptop-Support-Stylus-Tablet-Computer-Tablet-Laptop-i.360719776.20559085907?sp_atk=5b3c3e47-5461-467b-84c6-4ac099e6e258&xptdk=5b3c3e47-5461-467b-84c6-4ac099e6e258"
reviews = pd.read_csv(args.csv)

# Clean reviews remove reviews with no comments
df = reviews['comment'].to_frame()
df.replace('', np.nan, inplace=True)
# Get comment IDs for rating analysis later
comment_id = []
for idx, sentence in df.itertuples():
    if not pd.isna(sentence):
        comment_id.append(idx)
# Drop data without comments
df.dropna(inplace=True)
total_reviews_comments = df.shape[0]

# ========================Rating based (Those without comments)===========================================
rating_df = reviews['rating'].to_frame()
# Drop data with comments
rating_df.drop(comment_id,inplace=True)
# rating <4 considered bad
bad_rating = 0
good_rating = 0
for idx, rating in rating_df.itertuples():
    if float(rating) < 4:
        bad_rating += 1
    else:
        good_rating += 1


# ========================Run Sentient model to get pos reviews==============================
# Get general ratings based on aspects (quality, service, delivery)
aspects = ['quality', 'service', 'delivery']
input_data = []
for idx, sentence in df.itertuples():
    input_data.append({"aspects": aspects, "sentence":sentence})

# Create tokenizer
tokenizer = SenticGCNBertTokenizer.from_pretrained("bert-base-uncased")

# Create embedding model
embed_config = SenticGCNBertEmbeddingConfig.from_pretrained("bert-base-uncased")
embed_model = SenticGCNBertEmbeddingModel.from_pretrained("bert-base-uncased", config=embed_config)

# Create preprocessor
preprocessor = SenticGCNBertPreprocessor(
    tokenizer=tokenizer,
    embedding_model=embed_model,
    senticnet="https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticnet.pickle",
    device="cpu",
)

# Create postprocessor
postprocessor = SenticGCNBertPostprocessor()

# Load model
config = SenticGCNBertConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/config.json"
)
model = SenticGCNBertModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/pytorch_model.bin", config=config
)

# Run sentient architecture
# Preprocess
processed_inputs, processed_indices = preprocessor(input_data)
# GCN model
outputs = model(processed_indices)
# Postprocessing
post_outputs = postprocessor(processed_inputs=processed_inputs, model_outputs=outputs)

pos_reviews = []
other_reviews = []
for index in range(len(post_outputs)):

    sum = np.sum(np.array(post_outputs[index]['labels']))

    if sum > 0:
        pos_reviews.append({"aspects":aspects, "sentence":' '.join(post_outputs[index]['sentence'])})
    else:
        other_reviews.append(post_outputs[index])

# ==================================Filter off potential fake reviews=================================
review_data = [x['sentence'] for x in pos_reviews]
pos_reviews_df = pd.DataFrame(data=review_data)
# Detect fake reviews
fake_idx = isol_predict(input = pos_reviews_df[0].values.tolist())
# Drop fake reviews
pos_reviews_df.drop(fake_idx, inplace=True)

# ======================== Analyse filtered positive reviews ===========================
filtered_data = []
for idx, sentence in pos_reviews_df.itertuples():
    filtered_data.append({"aspects": aspects, "sentence":sentence})

if filtered_data == []:
    pass
else:
    # Preprocess
    filtered_processed_inputs, filtered_processed_indices = preprocessor(filtered_data)
    # GCN model
    filtered_outputs = model(filtered_processed_indices)
    # Postprocessing
    filtered_data = postprocessor(processed_inputs=filtered_processed_inputs, model_outputs=filtered_outputs)

# =========================== Compile data =======================================
pos_score = {'quality': 0,
         'service': 0,
         'delivery': 0}

# Compile for pos reviews
for x in filtered_data:
    # Match label and aspect
    for y in range(len(x['aspects'])):
        score_type = re.sub('\W+', '', x['sentence'][int(x['aspects'][y][0])].lower())

        pos_score[score_type] += int(x['labels'][y])

bad_score = {'quality': 0,
            'service': 0,
            'delivery': 0}
# Compile for bad reviews
for x in other_reviews:
    # Match label and aspect
    for y in range(len(x['aspects'])):
        score_type = re.sub('\W+', '', x['sentence'][int(x['aspects'][y][0])].lower())

        bad_score[score_type] += -int(x['labels'][y])

# Combine with rating scores
for key in bad_score.keys():
    bad_score[key] += bad_rating

for key in pos_score.keys():
    pos_score[key] += good_rating

# Results
quality = round((pos_score['quality']/(pos_score['quality']+bad_score['quality'])) *100)
service = round((pos_score['service']/(pos_score['service']+bad_score['service'])) *100)
delivery = round((pos_score['delivery']/(pos_score['delivery']+bad_score['delivery'])) *100)

print("===============RATINGS=================")
print("No. of potential fake reviews: ", len(fake_idx))
print("Quality:", quality,'%')
print("Service:", service, '%')
print("Delivery:", delivery, '%')


