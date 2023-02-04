from aspect_abstraction import get_aspects
import pickle
from sgnlp.models.sentic_gcn import (
    SenticGCNBertTokenizer,
    SenticGCNBertEmbeddingConfig,
    SenticGCNBertEmbeddingModel,
    SenticGCNBertModel,
    SenticGCNBertPreprocessor,
    SenticGCNBertConfig,
    SenticGCNBertPostprocessor,
)
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

def isol_preprocess(input_data):
    # Get aspect
    inputs = []
    for review in input_data:

        aspect_list = []
        aspect = get_aspects(review)
        aspect_list.append(aspect)
        inputs.append({"aspects":aspect, "sentence": review})

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
    postprocessor = SenticGCNBertPostprocessor(return_probabilities_before_argmax=True)

    # Load model
    config = SenticGCNBertConfig.from_pretrained(
        "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/config.json"
    )
    model = SenticGCNBertModel.from_pretrained(
        "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/pytorch_model.bin", config=config
    )

    # Run sentient architecture
    # Preprocess
    processed_inputs, processed_indices = preprocessor(inputs)
    # GCN model
    outputs = model(processed_indices)
    # Postprocessing
    post_outputs = postprocessor(processed_inputs=processed_inputs, model_outputs=outputs)
    # Get probability of each label
    prob = []
    for output in post_outputs:
        prob.append(output['probability'])

    return prob

def isol_trainer(input):
    # Isolation forest
    isol_model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
    res = isol_model.fit(input[0])

    # Save model
    with open('FakeReviewModel.pkl', 'wb') as f:
        pickle.dump(isol_model, f)

def isol_predict(input, model="FakeReviewModel.pkl"):
    proccessed_data = isol_preprocess(input)
    isol_model = pd.read_pickle(model)

    fake_index = []
    for index in range(len(proccessed_data)):
        res = isol_model.predict(proccessed_data[index])

        final_res = np.sum(res)
        if final_res < 0:
            # Fake review found
            fake_index.append(index)
        else:
            pass
    return fake_index

if __name__ == "__main__":
    # Test data
    # testData = [
    #     'Tried it and works well! Can mirror phone on TV screen easily',
    #     'Working well and delivery time is relatively fast from China',
    #     'Fast shipping great quality.. have yet to try will review again if anything goes wrong.',
    #     "Delivery was fast. Item received in good condition. Haven't tested the product yet. But have faith in product it will function as stated. ",
    #     "delivery was quite slow, I needed it urgently. but it was 11.11 so I Guess that’s the reason why? Cable works well, tried and tested. Thank you!"
    # ]
    testData = [
        # 'Tried it and works well! Can mirror phone on TV screen easily',
        # 'Working well and delivery time is relatively fast from China',
        # 'Fast shipping great quality.. have yet to try will review again if anything goes wrong.',
        "LOVE THIS calculator!!!  all the functions OMG~~~ one thing tho, make sure that your kids don't get it. "
        "it will ruin them. Only get it if you understand the math behind the functions that you are looking to use., "
        "if not, it makes life too easy. which, it is handy, but deters learning",

        "This product is perfect for what it is intended to do. If you have a larger head (or smaller body) you may want to "
        "get the TRS-80A2 instead. I had this for a few months, and it worked great. I used it for a couple of weeks.",

        # "I like thin hardshell cases, but I am very disappointed with the quality of the case. The case is too thick and the "
        # "plastic feels like it is too tight. I would definitely buy again. It will cost me a lot more. I bought this to "
        # "replace a broken one I bought for my TV. I did not want to have to replace it since the TV was so cheap. "
        # "It works fine and seems to last for quite a while."

        # "Delivery was fast. Item received in good condition. Haven't tested the product yet. But have faith in product it will function as stated. ",
        # "delivery was quite slow, I needed it urgently. but it was 11.11 so I Guess that’s the reason why? Cable works well, tried and tested. Thank you!"
    ]

    # Read input data
    input_data = pd.read_csv('data1.csv')
    input_data = input_data['comment']
    input_data = input_data.dropna()
    # preprocess data
    processed_input = isol_preprocess(input_data)

    # Train model
    isol_trainer(processed_input)

    # Run model
    res = isol_predict(testData)