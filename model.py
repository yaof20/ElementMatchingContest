from ernie.file_utils import add_docstring
import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D
from ernie.modeling_ernie import ErnieModel


def _build_linear(n_in, n_out, name, init, act=None):
    return D.Linear(n_in,
                    n_out,
                    param_attr=F.ParamAttr(name='%s.w_0' % name if name is not None else None, initializer=init),
                    bias_attr='%s.b_0' % name if name is not None else None, act=act)


def append_name(name, postfix):
    if name is None:
        return None
    elif name == '':
        return postfix
    else:
        return '%s_%s' % (name, postfix)


class ErnieForElementClassification(ErnieModel):
    def __init__(self, cfg, name=None):
        super(ErnieForElementClassification, self).__init__(cfg, name=name)

        initializer = F.initializer.TruncatedNormal(scale=cfg['initializer_range'])
        self.classifier = _build_linear(cfg['hidden_size'], cfg['num_labels'], append_name(name, 'cls'), initializer)  #

        prob = cfg.get('classifier_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = lambda i: L.dropout(i, dropout_prob=prob, dropout_implementation="upscale_in_train",) if self.training else i

    @add_docstring(ErnieModel.forward.__doc__)
    def forward(self, *args, **kwargs):
        """
        Args:
            22labels (optional, `Variable` of shape [batch_size]):
                ground truth label id for each sentence
        Returns:
            loss (`Variable` of shape []):
                Cross entropy loss mean over batch
                if labels not set, returns None
            logits (`Variable` of shape [batch_size, hidden_size]):
                output logits of classifier
        """
        labels = kwargs.pop('labels', None)
        pooled, encoded = super(ErnieForElementClassification, self).forward(*args, **kwargs)
        hidden = self.dropout(pooled)
        logits = self.classifier(hidden)
        logits = L.sigmoid(logits)
        sqz_logits = L.squeeze(logits, axes=[1])
        if labels is not None:
            if len(labels.shape) == 1:
                labels = L.reshape(labels, [-1, 1])
            part1 = L.elementwise_mul(labels, L.log(logits))
            part2 = L.elementwise_mul(1-labels, L.log(1-logits))
            loss = - L.elementwise_add(part1, part2)
            loss = L.reduce_mean(loss)
            return loss, sqz_logits
        else:
            return sqz_logits
