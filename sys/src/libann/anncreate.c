#include <u.h>
#include <libc.h>
#include <ann.h>
#define RAND_MAX 0xFFFF

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
