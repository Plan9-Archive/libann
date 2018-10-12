#include <u.h>
#include <libc.h>
#include <ann.h>

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
