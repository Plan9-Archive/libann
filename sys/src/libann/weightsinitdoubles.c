#include <u.h>
#include <libc.h>
#include <ann.h>

Weights*
weightsinitdoubles(Weights *in, double *init)
{
	int i, o;

	for (i = 0; i < in->inputs; i++)
		for (o = 0; o < in->outputs; o++)
			in->values[i][o] = init[o];

	return in;
}
