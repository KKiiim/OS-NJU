// Original Author: Andrej Karpathy
// https://github.com/karpathy/llm.c

#include "thread-sync.h"
#include "thread.h"
#include "types.h"
#include <assert.h>

// ----------------------------------------------------------------------------
// all the individual layers' forward passes
// BatchSize = batch_size, SeqLen = sequence_length, C = channels, V = vocab_size

void encoder_forward(float *out, int *inp, float *wte, float *wpe, int BatchSize, int SeqLen, int C) {
  // out is (BatchSize,SeqLen,C). At each position (b,t), a C-dimensional vector summarizing
  // token & position inp is (BatchSize,SeqLen) of integers, holding the token ids at each
  // (b,t) position wte is (V,C) of token embeddings, short for "weight token
  // embeddings" wpe is (maxT,C) of position embeddings, short for "weight
  // positional embedding"
  for (int b = 0; b < BatchSize; b++) {
    for (int t = 0; t < SeqLen; t++) {
      // seek to the output position in out[b,t,:]
      float *out_bt = out + b * SeqLen * C + t * C;
      // get the index of the token at inp[b, t]
      int ix = inp[b * SeqLen + t];
      // seek to the position in wte corresponding to the token
      float *wte_ix = wte + ix * C;
      // seek to the position in wpe corresponding to the position
      float *wpe_t = wpe + t * C;
      // add the two vectors and store the result in out[b,t,:]
      for (int i = 0; i < C; i++) {
        //
        // 1*3*768
        //
        out_bt[i] = wte_ix[i] + wpe_t[i];
      }
    }
  }
}

void layernorm_forward(float *out, float *mean, float *rstd, float *inp, float *weight, float *bias, int BatchSize,
                       int SeqLen, int C) {
  // reference:
  // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html both inp
  // and out are (BatchSize,SeqLen,C) of the activations mean and rstd are (BatchSize,SeqLen) buffers, to
  // be used later in backward pass at each position (b,t) of the input, the
  // C-dimensional vector of activations gets normalized, then scaled and
  // shifted
  float eps = 1e-5f;
  for (int b = 0; b < BatchSize; b++) {
    for (int t = 0; t < SeqLen; t++) {
      // seek to the input position inp[b,t,:]
      float *x = inp + b * SeqLen * C + t * C;
      // calculate the mean
      float m = 0.0f;
      for (int i = 0; i < C; i++) {
        //
        // 1*3*768
        //
        m += x[i];
      }
      m = m / C;
      // calculate the variance (without any bias correction)
      float v = 0.0f;
      for (int i = 0; i < C; i++) {
        //
        // 1*3*768
        //
        float xshift = x[i] - m;
        v += xshift * xshift;
      }
      v = v / C;
      // calculate the rstd (reciprocal standard deviation)
      float s = 1.0f / sqrtf(v + eps);
      // seek to the output position in out[b,t,:]
      float *out_bt = out + b * SeqLen * C + t * C;
      for (int i = 0; i < C; i++) {
        //
        // 1*3*768
        //
        float n = (s * (x[i] - m));        // normalize
        float o = n * weight[i] + bias[i]; // scale and shift
        out_bt[i] = o;                     // write
      }
      // cache the mean and rstd for the backward pass later
      mean[b * SeqLen + t] = m;
      rstd[b * SeqLen + t] = s;
    }
  }
}

typedef struct {
  int C;
  const float *a;
  const float *b;
  float result;
} MultiAndAccumParams;

static MultiAndAccumParams paramsArray[NumberOfThreads];

float multiAndAccum(const int C, float const *const a, float const *const b) {
  float res = 0.0f;
  for (int i = 0; i < C; i++) {
    res += a[i] * b[i];
  }
  return res;
}

void multiAndAccumWrapper(int id) {
  MultiAndAccumParams *params = &paramsArray[id - 1];
  params->result += multiAndAccum(params->C, params->a, params->b);
}

void matmul_forward(float *out, float *inp, float *weight, float *bias, int BatchSize, int SeqLen, int C, int OC) {
  // most of the running time is spent here and in matmul_backward
  // OC is short for "output channels"
  // inp is (BatchSize,SeqLen,C), weight is (OC, C), bias is (OC)
  // out will be (BatchSize,SeqLen,OC)
  for (int b = 0; b < BatchSize; b++) {
    for (int t = 0; t < SeqLen; t++) {
      float *out_bt = out + b * SeqLen * OC + t * OC;
      float *inp_bt = inp + b * SeqLen * C + t * C;

      int o = 0;
      for (; o + 3 < OC; o = o + 4) {
        for (int i = 0; i < NumberOfThreads; i++) {
          paramsArray[i].C = C;
          paramsArray[i].a = inp_bt;
          paramsArray[i].b = weight + (o + i) * C;
          paramsArray[i].result = (bias != NULL) ? bias[o + i] : 0.0f;
        }

        for (int i = 0; i < NumberOfThreads; i++) {
          create((void *)multiAndAccumWrapper);
        }

        // for (int i = 0; i < C; i++) {
        //   //
        //   // 1*3*2304*768
        //   //
        //   val += inp_bt[i] * wrow[i];
        // }

        join();

        for (int i = 0; i < NumberOfThreads; i++) {
          out_bt[o + i] += paramsArray[i].result;
        }
        resetThreads();
      }

      assert((OC - o) == (OC % 4));
      for (int remain = o; o < OC; o++) {
        float val0 = (bias != NULL) ? bias[o] : 0.0f;
        float *wrow0 = weight + o * C;
        val0 += multiAndAccum(C, inp_bt, wrow0);
        out_bt[o] = val0;
      }
    }
  }
}

void attention_forward(float *out, float *preatt, float *att, float *inp, int BatchSize, int SeqLen, int C, int NH) {
  // input is (BatchSize, SeqLen, 3C) holding the query, key, value (Q, K, V) vectors
  // preatt, att are (BatchSize, NH, SeqLen, SeqLen). NH = number of heads, SeqLen = sequence length
  // that holds the pre-attention and post-attention scores (used in backward)
  // output is (BatchSize, SeqLen, C)
  // attention is the only layer that mixes information across time
  // every other operation is applied at every (b,t) position independently
  // (and of course, no layer mixes information across batch)
  int C3 = C * 3;
  int hs = C / NH; // head size
  float scale = 1.0 / sqrtf(hs);

  for (int b = 0; b < BatchSize; b++) {
    for (int t = 0; t < SeqLen; t++) {
      for (int h = 0; h < NH; h++) {
        float *query_t = inp + b * SeqLen * C3 + t * C3 + h * hs;
        float *preatt_bth = preatt + b * NH * SeqLen * SeqLen + h * SeqLen * SeqLen + t * SeqLen;
        float *att_bth = att + b * NH * SeqLen * SeqLen + h * SeqLen * SeqLen + t * SeqLen;

        // pass 1: calculate query dot key and maxval
        float maxval = -10000.0f; // TODO something better
        for (int t2 = 0; t2 <= t; t2++) {
          float *key_t2 = inp + b * SeqLen * C3 + t2 * C3 + h * hs + C; // +C because it's key

          // (query_t) dot (key_t2)
          float val = 0.0f;
          for (int i = 0; i < hs; i++) {
            val += query_t[i] * key_t2[i];
          }
          val *= scale;
          if (val > maxval) {
            maxval = val;
          }

          preatt_bth[t2] = val;
        }

        // pass 2: calculate the exp and keep track of sum
        // maxval is being calculated and subtracted only for numerical
        // stability
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
          float expv = expf(preatt_bth[t2] - maxval);
          expsum += expv;
          att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // pass 3: normalize to get the softmax
        for (int t2 = 0; t2 < SeqLen; t2++) {
          if (t2 <= t) {
            att_bth[t2] *= expsum_inv;
          } else {
            // causal attention mask. not strictly necessary to set to zero here
            // only doing this explicitly for debugging and checking to PyTorch
            att_bth[t2] = 0.0f;
          }
        }

        // pass 4: accumulate weighted values into the output of attention
        float *out_bth = out + b * SeqLen * C + t * C + h * hs;
        for (int i = 0; i < hs; i++) {
          out_bth[i] = 0.0f;
        }
        for (int t2 = 0; t2 <= t; t2++) {
          float *value_t2 = inp + b * SeqLen * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
          float att_btht2 = att_bth[t2];
          for (int i = 0; i < hs; i++) {
            out_bth[i] += att_btht2 * value_t2[i];
          }
        }
      }
    }
  }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float *out, float *inp, int N) {
  // (approximate) GeLU elementwise non-linearity in the MLP block of
  // Transformer
  for (int i = 0; i < N; i++) {
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
  }
}

void residual_forward(float *out, float *inp1, float *inp2, int N) {
  for (int i = 0; i < N; i++) {
    out[i] = inp1[i] + inp2[i];
  }
}

void softmax_forward(float *probs, float *logits, int BatchSize, int SeqLen, int V) {
  // output: probs are (BatchSize,SeqLen,V) of the probabilities (sums to 1.0 in each b,t
  // position) input: logits is (BatchSize,SeqLen,V) of the unnormalized log probabilities
  for (int b = 0; b < BatchSize; b++) {
    for (int t = 0; t < SeqLen; t++) {
      // probs <- softmax(logits)
      float *logits_bt = logits + b * SeqLen * V + t * V;
      float *probs_bt = probs + b * SeqLen * V + t * V;

      // maxval is only calculated and subtracted for numerical stability
      float maxval = -10000.0f; // TODO something better
      for (int i = 0; i < V; i++) {
        if (logits_bt[i] > maxval) {
          maxval = logits_bt[i];
        }
      }
      float sum = 0.0f;
      for (int i = 0; i < V; i++) {
        probs_bt[i] = expf(logits_bt[i] - maxval);
        sum += probs_bt[i];
      }
      for (int i = 0; i < V; i++) {
        probs_bt[i] /= sum;
      }
    }
  }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

// allocate memory for the parameters and point the individual tensors to the
// right places
float *malloc_and_point_parameters(ParameterTensors *params, size_t *param_sizes) {
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += param_sizes[i];
  }
  // malloc all parameters all at once
  float *params_memory = (float *)malloc(num_parameters * sizeof(float));
  // assign all the tensors
  float **ptrs[] = {&params->wte,      &params->wpe,      &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
                    &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw,  &params->fcb,
                    &params->fcprojw,  &params->fcprojb,  &params->lnfw, &params->lnfb};
  float *params_memory_iterator = params_memory;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    *(ptrs[i]) = params_memory_iterator;
    params_memory_iterator += param_sizes[i];
  }
  return params_memory;
}

float *malloc_and_point_activations(ActivationTensors *acts, size_t *act_sizes) {
  size_t num_activations = 0;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    num_activations += act_sizes[i];
  }
  float *acts_memory = (float *)malloc(num_activations * sizeof(float));
  float **ptrs[] = {&acts->encoded, &acts->ln1,       &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv,
                    &acts->atty,    &acts->preatt,    &acts->att,      &acts->attproj,  &acts->residual2,
                    &acts->ln2,     &acts->ln2_mean,  &acts->ln2_rstd, &acts->fch,      &acts->fch_gelu,
                    &acts->fcproj,  &acts->residual3, &acts->lnf,      &acts->lnf_mean, &acts->lnf_rstd,
                    &acts->logits,  &acts->probs,     &acts->losses};
  float *acts_memory_iterator = acts_memory;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    *(ptrs[i]) = acts_memory_iterator;
    acts_memory_iterator += act_sizes[i];
  }
  return acts_memory;
}

void gpt2_build_from_checkpoint(GPT2 *model, char *checkpoint_path) {

  // read in model from a checkpoint file
  FILE *model_file = fopen(checkpoint_path, "rb");
  if (model_file == NULL) {
    printf("Error opening model file\n");
    exit(1);
  }
  int model_header[256];
  fread(model_header, sizeof(int), 256, model_file);
  if (model_header[0] != 20240326) {
    printf("Bad magic model file");
    exit(1);
  }
  if (model_header[1] != 1) {
    printf("Bad version in model file");
    exit(1);
  }

  // read in hyperparameters
  int maxT, V, L, NH, C;
  model->config.max_seq_len = maxT = model_header[2];
  model->config.vocab_size = V = model_header[3];
  model->config.num_layers = L = model_header[4];
  model->config.num_heads = NH = model_header[5];
  model->config.channels = C = model_header[6];

  // allocate space for all the parameters and read them in
  model->param_sizes[0] = V * C;            // wte
  model->param_sizes[1] = maxT * C;         // wpe
  model->param_sizes[2] = L * C;            // ln1w
  model->param_sizes[3] = L * C;            // ln1b
  model->param_sizes[4] = L * (3 * C) * C;  // qkvw
  model->param_sizes[5] = L * (3 * C);      // qkvb
  model->param_sizes[6] = L * C * C;        // attprojw
  model->param_sizes[7] = L * C;            // attprojb
  model->param_sizes[8] = L * C;            // ln2w
  model->param_sizes[9] = L * C;            // ln2b
  model->param_sizes[10] = L * (4 * C) * C; // fcw
  model->param_sizes[11] = L * (4 * C);     // fcb
  model->param_sizes[12] = L * C * (4 * C); // fcprojw
  model->param_sizes[13] = L * C;           // fcprojb
  model->param_sizes[14] = C;               // lnfw
  model->param_sizes[15] = C;               // lnfb

  // cound the number of paramaters
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += model->param_sizes[i];
  }
  model->num_parameters = num_parameters;

  // read in all the parameters from file
  model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
  fread(model->params_memory, sizeof(float), num_parameters, model_file);
  fclose(model_file);

  // other inits
  model->acts_memory = NULL;
  model->grads_memory = NULL;
  model->m_memory = NULL;
  model->v_memory = NULL;
  model->grads_acts_memory = NULL;
  model->inputs = NULL;
  model->targets = NULL;
  model->batch_size = 0;
  model->seq_len = 0;
  model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void gpt2_forward(GPT2 *model, int *inputs, int BatchSize, int SeqLen) {
  // convenience parameters
  int V = model->config.vocab_size;
  int L = model->config.num_layers;
  int NH = model->config.num_heads;
  int C = model->config.channels;

  // record the current BatchSize,SeqLen as well
  model->batch_size = BatchSize;
  model->seq_len = SeqLen;
  // and now allocate the space
  model->act_sizes[0] = BatchSize * SeqLen * C;               // encoded
  model->act_sizes[1] = L * BatchSize * SeqLen * C;           // ln1
  model->act_sizes[2] = L * BatchSize * SeqLen;               // ln1_mean
  model->act_sizes[3] = L * BatchSize * SeqLen;               // ln1_rstd
  model->act_sizes[4] = L * BatchSize * SeqLen * 3 * C;       // qkv
  model->act_sizes[5] = L * BatchSize * SeqLen * C;           // atty
  model->act_sizes[6] = L * BatchSize * NH * SeqLen * SeqLen; // preatt
  model->act_sizes[7] = L * BatchSize * NH * SeqLen * SeqLen; // att
  model->act_sizes[8] = L * BatchSize * SeqLen * C;           // attproj
  model->act_sizes[9] = L * BatchSize * SeqLen * C;           // residual2
  model->act_sizes[10] = L * BatchSize * SeqLen * C;          // ln2
  model->act_sizes[11] = L * BatchSize * SeqLen;              // ln2_mean
  model->act_sizes[12] = L * BatchSize * SeqLen;              // ln2_rstd
  model->act_sizes[13] = L * BatchSize * SeqLen * 4 * C;      // fch
  model->act_sizes[14] = L * BatchSize * SeqLen * 4 * C;      // fch_gelu
  model->act_sizes[15] = L * BatchSize * SeqLen * C;          // fcproj
  model->act_sizes[16] = L * BatchSize * SeqLen * C;          // residual3
  model->act_sizes[17] = BatchSize * SeqLen * C;              // lnf
  model->act_sizes[18] = BatchSize * SeqLen;                  // lnf_mean
  model->act_sizes[19] = BatchSize * SeqLen;                  // lnf_rstd
  model->act_sizes[20] = BatchSize * SeqLen * V;              // logits
  model->act_sizes[21] = BatchSize * SeqLen * V;              // probs
  model->act_sizes[22] = BatchSize * SeqLen;                  // losses
  size_t num_activations = 0;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    num_activations += model->act_sizes[i];
  }
  model->num_activations = num_activations;

  if (model->acts_memory) {
    free(model->acts_memory);
    model->acts_memory = NULL;
  }
  model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);

  // also create memory for caching inputs and targets
  if (model->inputs) {
    free(model->inputs);
  }
  model->inputs = (int *)malloc(BatchSize * SeqLen * sizeof(int));

  // cache the inputs/targets
  memcpy(model->inputs, inputs, BatchSize * SeqLen * sizeof(int));

  // forward pass
  ParameterTensors params = model->params; // for brevity
  ActivationTensors acts = model->acts;
  float *residual;
  encoder_forward(acts.encoded, inputs, params.wte, params.wpe, BatchSize, SeqLen,
                  C); // encoding goes into residual[0]
  for (int l = 0; l < L; l++) {

    residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * BatchSize * SeqLen * C;

    // get the pointers of the weights for this layer
    float *l_ln1w = params.ln1w + l * C;
    float *l_ln1b = params.ln1b + l * C;
    float *l_qkvw = params.qkvw + l * 3 * C * C;
    float *l_qkvb = params.qkvb + l * 3 * C;
    float *l_attprojw = params.attprojw + l * C * C;
    float *l_attprojb = params.attprojb + l * C;
    float *l_ln2w = params.ln2w + l * C;
    float *l_ln2b = params.ln2b + l * C;
    float *l_fcw = params.fcw + l * 4 * C * C;
    float *l_fcb = params.fcb + l * 4 * C;
    float *l_fcprojw = params.fcprojw + l * C * 4 * C;
    float *l_fcprojb = params.fcprojb + l * C;

    // get the pointers of the activations for this layer
    float *l_ln1 = acts.ln1 + l * BatchSize * SeqLen * C;
    float *l_ln1_mean = acts.ln1_mean + l * BatchSize * SeqLen;
    float *l_ln1_rstd = acts.ln1_rstd + l * BatchSize * SeqLen;
    float *l_qkv = acts.qkv + l * BatchSize * SeqLen * 3 * C;
    float *l_atty = acts.atty + l * BatchSize * SeqLen * C;
    float *l_preatt = acts.preatt + l * BatchSize * NH * SeqLen * SeqLen;
    float *l_att = acts.att + l * BatchSize * NH * SeqLen * SeqLen;
    float *l_attproj = acts.attproj + l * BatchSize * SeqLen * C;
    float *l_residual2 = acts.residual2 + l * BatchSize * SeqLen * C;
    float *l_ln2 = acts.ln2 + l * BatchSize * SeqLen * C;
    float *l_ln2_mean = acts.ln2_mean + l * BatchSize * SeqLen;
    float *l_ln2_rstd = acts.ln2_rstd + l * BatchSize * SeqLen;
    float *l_fch = acts.fch + l * BatchSize * SeqLen * 4 * C;
    float *l_fch_gelu = acts.fch_gelu + l * BatchSize * SeqLen * 4 * C;
    float *l_fcproj = acts.fcproj + l * BatchSize * SeqLen * C;
    float *l_residual3 = acts.residual3 + l * BatchSize * SeqLen * C;

    // now do the forward pass
    layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, BatchSize, SeqLen, C);
    matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, BatchSize, SeqLen, C, 3 * C);
    attention_forward(l_atty, l_preatt, l_att, l_qkv, BatchSize, SeqLen, C, NH);
    matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, BatchSize, SeqLen, C, C);
    residual_forward(l_residual2, residual, l_attproj, BatchSize * SeqLen * C);
    layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, BatchSize, SeqLen, C);
    matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, BatchSize, SeqLen, C, 4 * C);
    gelu_forward(l_fch_gelu, l_fch, BatchSize * SeqLen * 4 * C);
    matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, BatchSize, SeqLen, 4 * C, C);
    residual_forward(l_residual3, l_residual2, l_fcproj, BatchSize * SeqLen * C);
  }
  residual = acts.residual3 + (L - 1) * BatchSize * SeqLen * C; // last residual is in residual3
  layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, BatchSize, SeqLen, C);
  matmul_forward(acts.logits, acts.lnf, params.wte, NULL, BatchSize, SeqLen, C, V);
  softmax_forward(acts.probs, acts.logits, BatchSize, SeqLen, V);
}

void gpt2_zero_grad(GPT2 *model) {
  if (model->grads_memory != NULL) {
    memset(model->grads_memory, 0, model->num_parameters * sizeof(float));
  }
  if (model->grads_acts_memory != NULL) {
    memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float));
  }
}

void gpt2_free(GPT2 *model) {
  free(model->params_memory);
  free(model->grads_memory);
  free(model->m_memory);
  free(model->v_memory);
  free(model->acts_memory);
  free(model->grads_acts_memory);
  free(model->inputs);
  free(model->targets);
}

int sample_mult(float *probabilities, int n) {
  // sample index from probabilities (they must sum to 1!)
  // coin can be a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f, coin = 0.5f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

// the GPT-2 end-of-text token id
#define GPT2_EOT 50256

int main(int argc, char **argv) {
  GPT2 model;
  gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
  const int maxTokens = 100; // Token limit.

  if (argc == 1) {
    printf("Provide at least one token.\n");
    exit(1);
  }
  if (argc > maxTokens) {
    printf("Tow many tokens.\n");
    exit(1);
  }

  int tokens[maxTokens];
  for (int i = 0; i < maxTokens; i++) {
    if (i + 1 < argc) {
      tokens[i] = strtol(argv[i + 1], NULL, 10);
    } else {
      tokens[i] = GPT2_EOT;
    }
  }

  const int batchSize = 1; // the batch size (BatchSize) of current forward pass指一次输入到模型的样本数量
  // workTokenIndex init by last input token index
  for (int workTokenIndex = argc - 1; workTokenIndex < maxTokens; workTokenIndex++) {
    gpt2_forward(&model, tokens, batchSize, workTokenIndex);
    float *probs = model.acts.probs + (workTokenIndex - 1) * model.config.vocab_size;
    int next_token = sample_mult(probs, model.config.vocab_size);
    tokens[workTokenIndex] = next_token;

    printf("%d\n", tokens[workTokenIndex]);
    fflush(stdout);
  }

  gpt2_free(&model);

  return 0;
}
