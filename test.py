from typing import Dict, List, Union

import torch.nn.functional as F

from .preprocess import SenticGCNData, SenticGCNBertData
from .modeling import SenticGCNModelOutput, SenticGCNBertModelOutput


class SenticGCNBasePostprocessor:
    """
    Base postprocessor class providing common post processing functions.
    """

    def __init__(self, return_full_text: bool = False, return_aspects_text: bool = False) -> None:
        self.return_full_text = return_full_text
        self.return_aspects_text = return_aspects_text

    def __call__(
        self,
        processed_inputs: List[Union[SenticGCNData, SenticGCNBertData]],
        model_outputs: Union[SenticGCNModelOutput, SenticGCNBertModelOutput],
    ) -> List[Dict[str, Union[List[str], List[int], float]]]:
        # Get predictions
        probabilities = F.softmax(model_outputs.logits, dim=-1).detach().numpy()
        predictions = [probabilities.argmax(axis=-1)[idx] - 1 for idx in range(len(probabilities))]
        # Process output
        outputs = []
        for processed_input, prediction in zip(processed_inputs, predictions):
            exists = False
            # Check to see if the full_text_tokens already exists
            # If found, append the aspect_token_index, prediction and optionally aspect texts.
            for idx, proc_output in enumerate(outputs):
                if proc_output["sentence"] == processed_input.full_text_tokens:
                    exists = True
                    outputs[idx]["aspects"].append(processed_input.aspect_token_indexes)
                    outputs[idx]["labels"].append(int(prediction))
                    if self.return_aspects_text:
                        outputs[idx]["aspects_text"].append(processed_input.aspect)
                    break
            if exists:
                continue
            processed_dict = {}
            processed_dict["sentence"] = processed_input.full_text_tokens
            processed_dict["aspects"] = [processed_input.aspect_token_indexes]
            processed_dict["labels"] = [int(prediction)]
            if self.return_full_text:
                processed_dict["full_text"] = processed_input.full_text
            if self.return_aspects_text:
                processed_dict["aspects_text"] = [processed_input.aspect]
            outputs.append(processed_dict)
        return outputs


class SenticGCNPostprocessor(SenticGCNBasePostprocessor):
    """
    Class to initialise the Postprocessor for SenticGCNModel.
    Class to postprocess SenticGCNModel output to get a list of input text tokens,
    aspect token index and prediction labels.

    Args:
        return_full_text (bool): Flag to indicate if the full text should be included in the output.
        return_aspects_text (bool): Flag to indicate if the list of aspects text should be included in the output.
    """

    def __init__(self, return_full_text: bool = False, return_aspects_text: bool = False) -> None:
        super().__init__(return_full_text=return_full_text, return_aspects_text=return_aspects_text)


class SenticGCNBertPostprocessor(SenticGCNBasePostprocessor):
    """
    Class to initialise the Postprocessor for SenticGCNBertModel.
    Class to postprocess SenticGCNBertModel output to get a list of input text tokens,
    aspect token index and prediction labels.

    Args:
        return_full_text (bool): Flag to indicate if the full text should be included in the output.
        return_aspects_text (bool): Flag to indicate if the list of aspects text should be included in the output.
    """

    def __init__(self, return_full_text: bool = False, return_aspects_text: bool = False) -> None:
        super().__init__(return_full_text=return_full_text, return_aspects_text=return_aspects_text)


from aspect_abstraction import get_aspects
from sgnlp.models.sentic_gcn import (
    SenticGCNBertTokenizer,
    SenticGCNBertEmbeddingConfig,
    SenticGCNBertEmbeddingModel,
    SenticGCNBertModel,
    SenticGCNBertPreprocessor,
    SenticGCNBertConfig,
   # SenticGCNBertPostprocessor,
)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


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
    "Good product",
    # "Delivery was fast. Item received in good condition. Haven't tested the product yet. But have faith in product it will function as stated. ",
    # "delivery was quite slow, I needed it urgently. but it was 11.11 so I Guess that’s the reason why? Cable works well, tried and tested. Thank you!"
]


# Get aspect
inputs = []
for review in testData:
    aspect_list = []
    aspect = get_aspects(review)
    aspect_list.append(aspect)
    print("aspect: ", aspect)
    inputs.append({"aspects":aspect, "sentence": review})

print(inputs)

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

model.gc3.register_forward_hook(get_activation("gc3"))



# Inputs
# inputs = [
#     {  # Single word aspect
#         "aspects": aspect,
#         #"sentence": "To sum it up : service varies from good to mediorce , depending on which waiter you get ; generally it is just average ok .",
#         "sentence": "The service is ok.",
#     },
#     # {  # Single-word, multiple aspects
#     #     "aspects": ["service", "decor"],
#     #     "sentence": "Everything is always cooked to perfection , the service is excellent, the decor cool and understated.",
#     # },
#     # {  # Multi-word aspect
#     #     "aspects": ["grilled chicken", "chicken"],
#     #     "sentence": "the only chicken i moderately enjoyed was their grilled chicken special with edamame puree .",
#     # },
# ]

processed_inputs, processed_indices = preprocessor(inputs)
print(processed_inputs)
# print(processed_indices)

outputs = model(processed_indices)


# var = model
# print('dir: ', dir(model.modules))
# print(var)
# print(type(var))
print("Model: ", model)
print("Ello?", activation['gc3'])

print("Type: ", type(outputs))
print("OUTPUTS: ", outputs)

print("processed_inputs: ", processed_inputs)

# Postprocessing
post_outputs = postprocessor(processed_inputs=processed_inputs, model_outputs=outputs)
print("post_outputs: ", post_outputs)