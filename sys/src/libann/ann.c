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
