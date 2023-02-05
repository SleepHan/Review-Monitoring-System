from sgnlp.models.sentic_gcn import(
    SenticGCNBertConfig,
    SenticGCNBertModel,
    SenticGCNBertEmbeddingConfig,
    SenticGCNBertEmbeddingModel,
    SenticGCNBertTokenizer,
    SenticGCNBertPreprocessor,
    SenticGCNBertPostprocessor
)
import json

# Test data, to be chamged to reading csv file
testData = [
    'Tried it and works well! Can mirror phone on TV screen easily', 
    'Working well and delivery time is relatively fast from China',
    'Fast shipping great quality.. have yet to try will review again if anything goes wrong.',
    "Delivery was fast. Item received in good condition. Haven't tested the product yet. But have faith in product it will function as stated. ",
    "delivery was quite slow, I needed it urgently. but it was 11.11 so I Guess that’s the reason why? Cable works well, tried and tested. Thank you!"
]


# Model config stuff
tokenizer = SenticGCNBertTokenizer.from_pretrained("bert-base-uncased")

config = SenticGCNBertConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/config.json"
)

model = SenticGCNBertModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/pytorch_model.bin",
    config=config
)

embed_config = SenticGCNBertEmbeddingConfig.from_pretrained("bert-base-uncased")

embed_model = SenticGCNBertEmbeddingModel.from_pretrained("bert-base-uncased",
    config=embed_config
)

preprocessor = SenticGCNBertPreprocessor(
    tokenizer=tokenizer, embedding_model=embed_model,
    senticnet="https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticnet.pickle",
    device="cpu")

postprocessor = SenticGCNBertPostprocessor()


finalInput = []

# Manual input of the different aspects for now
# Planning to change to generalize aspects with common meanings
# E.g.
#    delivery: delivery, shipping
#    quality: work, works, working, quality, condition
for review in testData:
    aspects = []

    if 'delivery' in review.lower():
        aspects.append('delivery')
    
    if 'work' in review.lower():
        aspects.append('work')

    if 'shipping' in review.lower():
        aspects.append('shipping')

    if 'quality' in review.lower():
        aspects.append('quality')
    
    if 'condition' in review.lower():
        aspects.append('condition')

    finalInput.append(
        {
            "aspects": aspects,
            "sentence": review
        }
    )


# inputAspect = input('Enter aspects: ')

# inputReview = input('Enter review: ')

# finalInput = [
#     {
#         "aspects": [inputAspect],
#         "sentence": inputReview
#     }
# ]


# Sending inputs to model to process and predict
processed_inputs, processed_indices = preprocessor(finalInput)
raw_outputs = model(processed_indices)

post_outputs = postprocessor(processed_inputs=processed_inputs, model_outputs=raw_outputs)


print('\n\n')

print(finalInput)

for output in post_outputs:
    print(output)


# Writing results to file, since a bit hard to see if just printing to console
with open('resOutput.txt', 'w') as f:
    f.write('Inputs to Model\n')
    for reviewInput in finalInput:
        f.write(json.dumps(reviewInput))
        f.write('\n')
    f.write('\nResults\n')
    for output in post_outputs:
        f.write(json.dumps(output))
        f.write('\n')