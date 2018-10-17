#include <u.h>
#include <libc.h>
#include <ann.h>

void
main()
{
	int i, counter = 0;;
	Ann *test = anncreate(3, 2, 16, 1);
	double inputs[4][2] = { { 1.0, 1.0 }, {1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}};
	double outputs[4] = { 0.0, 1.0, 1.0, 0.0 };
	double error = 1000;

	//test->rate = 4.0;

	while (error >= 0.001) {
		error = 0;
		for (i = 0; i < 4; i++)
			error += anntrain_adam(test, inputs[i], &outputs[i]);	

		counter++;
		if (counter % 10000 == 1)
			print("error: %f\n", error);
	}

	print("error: %f, done after %d epochs\n", error, counter);
}
