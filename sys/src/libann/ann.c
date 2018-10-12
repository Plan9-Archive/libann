#include <u.h>
#include <libc.h>
#include <ann.h>
#define RAND_MAX 0xFFFF

double
activation_sigmoid(double x)
{
	return 1.0/(1.0+exp(-x));
}

double
gradient_sigmoid(double x)
{
	double y = activation_sigmoid(x);
	return y * (1.0 - y);
}

Neuron*
neuroninit(Neuron *in, double (*activation)(double input), double (*gradient)(double input), double steepness)
{
	in->activation = activation;
	in->gradient = gradient;
	in->steepness = steepness;
	in->value = 0;
	in->sum = 0;
	return in;
}

Neuron*
neuroncreate(double (*activation)(double input), double (*gradient)(double input), double steepness)
{
	Neuron *ret = calloc(1, sizeof(Neuron));
	neuroninit(ret, activation, gradient, steepness);
	return ret;
}

Layer*
layercreate(int num_neurons, double(*activation)(double), double(*gradient)(double))
{
	Layer *ret = calloc(1, sizeof(Layer));
	int i;

	ret->n = num_neurons;
	ret->neurons = calloc(num_neurons, sizeof(Neuron*));
	for (i = 0; i < ret->n; i++) {
		ret->neurons[i] = neuroncreate(activation, gradient, 1.0);
	}
	return ret;
}

Weights*
weightsinitrand(Weights *in)
{
	int i, o;

	srand(time(0));
	for (i = 0; i < in->inputs; i++)
		for (o = 0; o < in->outputs; o++)
			in->values[i][o] = ((double)rand()/RAND_MAX) - 0.5;

	return in;
}

Weights*
weightsinitdoubles(Weights *in, double *init)
{
	int i, o;

	for (i = 0; i < in->inputs; i++)
		for (o = 0; o < in->outputs; o++)
			in->values[i][o] = init[o];

	return in;
}

Weights*
weightsinitdouble(Weights *in, double init)
{
	int i, o;

	for (i = 0; i < in->inputs; i++)
		for (o = 0; o < in->outputs; o++)
			in->values[i][o] = init;

	return in;
}

Weights*
weightscreate(int inputs, int outputs, int initialize)
{
	int i;
	Weights *ret = calloc(1, sizeof(Weights));
	ret->inputs = inputs;
	ret->outputs = outputs;
	ret->values = calloc(inputs, sizeof(double*));
	for (i = 0; i < inputs; i++)
		ret->values[i] = calloc(outputs, sizeof(double));
	if (initialize)
		weightsinitrand(ret);
	else
		weightsinitdouble(ret, 1.0);
	return ret;
}

Ann*
anncreate(int num_layers, ...)
{
	Ann *ret = calloc(1, sizeof(Ann));
	va_list args;
	int arg;
	int i;

	va_start(args, num_layers);
	ret->n = num_layers;
	ret->rate = 0.25;
	ret->layers = calloc(num_layers, sizeof(Layer*));
	ret->weights = calloc(num_layers-1, sizeof(Weights*));
	ret->deltas = calloc(num_layers-1, sizeof(Weights*));

	for (i = 0; i < num_layers; i++) {
		arg = va_arg(args, int);
		if (arg < 0 || arg > 1000000)
			arg = 0;
		ret->layers[i] = layercreate(arg, activation_sigmoid, gradient_sigmoid);
		if (i > 0) {
			ret->weights[i-1] = weightscreate(ret->layers[i-1]->n, ret->layers[i]->n, 1);
			ret->deltas[i-1] = weightscreate(ret->layers[i-1]->n, ret->layers[i]->n, 0);
		}
	}	va_end(args);

	return ret;
}

double*
annrun(Ann *ann, double *input)
{
	int l, i, o;
	int outputs = ann->layers[ann->n - 1]->n;
	double *ret = calloc(outputs, sizeof(double));
	Neuron *O;

	for (i = 0; i < ann->layers[0]->n; i++)
		ann->layers[0]->neurons[i]->value = input[i];

	for (l = 1; l < ann->n; l++) {
		for (o = 0; o < ann->layers[l]->n; o++) {
			O = ann->layers[l]->neurons[o];
			O->sum = 0;
			for (i = 0; i < ann->layers[l-1]->n; i++)
				O->sum += ann->layers[l-1]->neurons[i]->value * ann->weights[l-1]->values[i][o];
			O->value = O->activation(O->sum);
		}
	}

	for (o = 0; o < outputs; o++)
		ret[o] = ann->layers[ann->n - 1]->neurons[o]->value;

	return ret;
}

void
anntrain(Ann *ann, double *inputs, double *outputs)
{
	double *error = annrun(ann, inputs);
	int noutputs = ann->layers[ann->n-1]->n;
	double acc, sum;
	int o, i, w, n;
	Neuron *O, *I;
	Weights *W, *D, *D2;

	for (o = 0; o < noutputs; o++) {
		// error = outputs[o] - result
		error[o] -= outputs[o];
		error[o] = -error[o];
	}
	D = ann->deltas[ann->n-2];
	weightsinitdoubles(D, error);

	// backpropagate MSE
	D2 = ann->deltas[ann->n-2];
	for (w = ann->n-2; w >= 0; w--) {
		D = ann->deltas[w];

		for (o = 0; o < ann->layers[w+1]->n; o++) {
			O = ann->layers[w+1]->neurons[o];
			acc = O->gradient(O->sum) * O->steepness;
			sum = 1.0;
			if (D2 != D) {
				W = ann->weights[w + 1];
				sum = 0.0;
				for (n = 0; n < D2->outputs; n++)
					sum += D2->values[o][n] * W->values[o][n];
			}
			for (i = 0; i < ann->layers[w]->n; i++) {
			 	D->values[i][o] *= acc * sum;
			}
		}

		D2 = D;
	}

	// update weights
	for (w = 0; w < ann->n-1; w++) {
		W = ann->weights[w];
		D = ann->deltas[w];

		for (i = 0; i < W->inputs; i++) {
			I = ann->layers[w]->neurons[i];
			for (o = 0; o < W->outputs; o++) {
				W->values[i][o] += D->values[i][o] * ann->rate * I->value;
			}
		}
	}

	free(error);
}
