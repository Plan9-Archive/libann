#include <u.h>
#include <libc.h>
#include <ann.h>

double
activation_sigmoid(Neuron *in)
{
	return 1.0/(1.0+exp(-in->sum));
}

double
gradient_sigmoid(Neuron *in)
{
	double y = in->value;
	return y * (1.0 - y);
}

double
activation_tanh(Neuron *in)
{
	return tanh(in->sum);
}

double
gradient_tanh(Neuron *in)
{
	return 1.0 - in->value*in->value;
}

double
activation_leaky_relu(Neuron *in)
{
	if (in->sum > 0)
		return in->sum;
	return in->sum * 0.01;
}

double
gradient_leaky_relu(Neuron *in)
{
	if (in->sum > 0)
		return 1.0;
	return 0.01;
}
