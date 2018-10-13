#include <u.h>
#include <libc.h>
#include <ann.h>
#define RAND_MAX 0xFFFF


Weights*
weightsinitdoubles(Weights *in, double *init)
{
	int i, o;

	for (i = 0; i <= in->inputs; i++)
		for (o = 0; o < in->outputs; o++)
			in->values[i][o] = init[o];

	return in;
}

Weights*
weightsinitdouble(Weights *in, double init)
{
	int i, o;

	for (i = 0; i <= in->inputs; i++)
		for (o = 0; o < in->outputs; o++)
			in->values[i][o] = init;

	return in;
}

Weights*
weightsinitrandscale(Weights *in, double scale)
{
	int i, o;

	srand(time(0));
	for (i = 0; i <= in->inputs; i++)
		for (o = 0; o < in->outputs; o++)
			in->values[i][o] = (((double)rand()/RAND_MAX) - 0.5) * scale;

	return in;
}

Weights*
weightsinitrand(Weights *in)
{
	weightsinitrandscale(in, 4.0);
	return in;
}
