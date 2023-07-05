#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor as T
from transformers import BertForMaskedLM

from .configuration_cxrbert import CXRBertConfig

BERTTupleOutput = Tuple[T, T, T, T, T]

class CXRBertOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    logits: torch.FloatTensor
    cls_projected_embedding: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertProjectionHead(nn.Module):
    '''
    Projection head to be used with BERT CLS token, it's similar to `BertPredictionHeadTransform` in HuggingFace library.
    :param config: CXRBertConfig
    :return: (batch_size, output_size)
    '''
    def __init__(self, config: CXRBertConfig) -> None:
        super().__init__()
        self.dense_to_hidden = nn.Linear(config.hidden_size, config.projection_size)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(config.projection_size, eps=1e-12)
        self.dense_to_output = nn.Linear(config.projection_size, config.projection_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_to_hidden(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense_to_output(hidden_states)

        return hidden_states


class CXRBertModel(BertForMaskedLM):
    """
    Implements the CXR-BERT model outlined in the manuscript:
    Boecking et al. "Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing", 2022
    https://arxiv.org/abs/2204.09817
    Extends the HuggingFace BertForMaskedLM model by adding a separate projection head. The projection "[CLS]" token is used to align
    the latent vectors of image and text modalities.
    """

    config_class = CXRBertConfig

    def __init__(self, config: CXRBertConfig):
        super().__init__(config)

        self.cls_projection_head = BertProjectionHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_cls_projected_embedding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[BERTTupleOutput, CXRBertOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bert_for_masked_lm_output = super().forward(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids,
                                                    position_ids=position_ids,
                                                    head_mask=head_mask,
                                                    inputs_embeds=inputs_embeds,
                                                    output_attentions=output_attentions,
                                                    output_hidden_states=True,
                                                    return_dict=True)

        last_hidden_state = bert_for_masked_lm_output.hidden_states[-1]
        cls_projected_embedding = self.cls_projection_head(last_hidden_state[:, 0, :]) if output_cls_projected_embedding else None

        if return_dict:
            return CXRBertOutput(
                last_hidden_state=last_hidden_state,
                logits=bert_for_masked_lm_output.logits,
                cls_projected_embedding=cls_projected_embedding,
                hidden_states=bert_for_masked_lm_output.hidden_states if output_hidden_states else None,
                attentions=bert_for_masked_lm_output.attentions,
            )
        else:
            return (
                last_hidden_state,
                bert_for_masked_lm_output.logits,
                cls_projected_embedding,
                bert_for_masked_lm_output.hidden_states,
                bert_for_masked_lm_output.attentions,)

    def get_projected_text_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns l2-normalised projected cls token embeddings for the given input token ids and attention mask.
        The joint latent space is trained using a contrastive objective between image and text data modalities.
        :param input_ids: (batch_size, sequence_length)
        :param attention_mask: (batch_size, sequence_length)
        :return: (batch_size, projection_size)
        """

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask,
                               output_cls_projected_embedding=True, return_dict=True)
        assert isinstance(outputs, CXRBertOutput)

        normalized_cls_embedding = F.normalize(outputs.cls_projected_embedding, dim=1)
        return normalized_cls_embedding

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generic utilities
"""

from collections import OrderedDict
from dataclasses import fields
from typing import Any, Tuple

import numpy as np

def is_tensor(x):
    """
    Tests if `x` is a `torch.Tensor`, `tf.Tensor`, `jaxlib.xla_extension.DeviceArray` or `np.ndarray`.
    """
    import torch

    if isinstance(x, torch.Tensor):
        return True

    return isinstance(x, np.ndarray)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())
