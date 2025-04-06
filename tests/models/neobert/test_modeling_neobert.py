import os
import unittest

from packaging import version

from transformers import AutoTokenizer, NeoBertConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        NeoBertForMaskedLM,
        NeoBertForSequenceClassification,
        NeoBertModel,
    )


class NeoBertModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        pad_token_id=0,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        embedding_init_range=0.02,
        decoder_init_range=0.02,
        norm_eps=1e-06,
        max_length=512,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.norm_eps = norm_eps
        self.max_length = max_length
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        """
        Returns a tiny configuration by default.
        """
        config = NeoBertConfig(
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_length=self.max_length,
            embedding_init_range=self.embedding_init_range,
            decoder_init_range=self.decoder_init_range,
            norm_eps=self.norm_eps,
        )
        if test := os.environ.get("PYTEST_CURRENT_TEST", False):
            test_name = test.split(":")[-1].split(" ")[0]

            # If we're testing `test_retain_grad_hidden_states_attentions`, we normally get an error
            # that compilation doesn't work. Users can then set compile=False when loading the model,
            # much like here. We're testing whether it works once they've done that.

            # If we're testing `test_inputs_embeds_matches_input_ids`, then we'd like to test with `reference_compile`
            # set to False, otherwise the input_ids with compiled input embeddings will not match the inputs_embeds
            # with atol=1e-8 and rtol=1e-5
            if test_name in ("test_retain_grad_hidden_states_attentions", "test_inputs_embeds_matches_input_ids"):
                config.reference_compile = False
            # Some tests require attentions to be outputted, in that case we'll set the attention implementation to eager
            # as the others don't support outputted attentions
            if test_name in (
                "test_attention_outputs",
                "test_hidden_states_output",
                "test_retain_grad_hidden_states_attentions",
            ):
                config._attn_implementation = "eager"
        return config

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = NeoBertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = NeoBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = NeoBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class NeoBertModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            NeoBertModel,
            NeoBertForMaskedLM,
            NeoBertForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": NeoBertModel,
            "fill-mask": NeoBertForMaskedLM,
            "text-classification": NeoBertForSequenceClassification,
            "zero-shot": NeoBertForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_head_masking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = NeoBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NeoBertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "chandar-lab/NeoBERT"
        model = NeoBertModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_torch
class NeoBertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        model = NeoBertForMaskedLM.from_pretrained("chandar-lab/NeoBERT")
        tokenizer = AutoTokenizer.from_pretrained("chandar-lab/NeoBERT")

        inputs = tokenizer("Hello World!", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 5, 50368))
        self.assertEqual(output.shape, expected_shape)

        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[3.8387, -0.2017, 12.2839], [3.6300, 0.6869, 14.7123], [-5.1137, -3.8122, 11.9874]]]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_no_head(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        model = NeoBertModel.from_pretrained("chandar-lab/NeoBERT")
        tokenizer = AutoTokenizer.from_pretrained("chandar-lab/NeoBERT")

        inputs = tokenizer("Hello World!", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 5, 768))
        self.assertEqual(output.shape, expected_shape)

        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[0.3151, -0.6417, -0.7027], [-0.7834, -1.5810, 0.4576], [1.0614, -0.7268, -0.0871]]]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_sequence_classification(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        model = NeoBertForSequenceClassification.from_pretrained("chandar-lab/NeoBERT")
        tokenizer = AutoTokenizer.from_pretrained("chandar-lab/NeoBERT")

        inputs = tokenizer("Hello World!", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 2))
        self.assertEqual(output.shape, expected_shape)

        expected = torch.tensor([[1.6466, 4.5662]])
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    @slow
    def test_export(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        bert_model = "chandar-lab/NeoBERT"
        max_length = 512

        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        inputs = tokenizer(
            "the man worked as a [MASK].",
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
        )

        model = NeoBertForMaskedLM.from_pretrained(
            bert_model,
        )

        logits = model(**inputs).logits
        eg_predicted_mask = tokenizer.decode(logits[0, 6].topk(5).indices)
        self.assertEqual(eg_predicted_mask.split(), ["lawyer", "mechanic", "teacher", "doctor", "waiter"])

        exported_program = torch.export.export(
            model,
            args=(inputs["input_ids"],),
            kwargs={"attention_mask": inputs["attention_mask"]},
            strict=True,
        )

        result = exported_program.module().forward(inputs["input_ids"], inputs["attention_mask"])
        ep_predicted_mask = tokenizer.decode(result.logits[0, 6].topk(5).indices)
        self.assertEqual(eg_predicted_mask, ep_predicted_mask)
