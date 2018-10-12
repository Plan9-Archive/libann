#include <u.h>
#include <libc.h>
#include <ann.h>

void
main()
{
	int i, j, counter = 0;;
	Ann *test = anncreate(3, 2, 16, 1);
	test->rate = 4.0;
	double inputs[4][2] = { { 1.0, 1.0 }, {1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}};
	double outputs[4] = { 0.0, 1.0, 1.0, 0.0 };
	double *results;
	double error = 1000;
	while (error > 0.001) {
		for (i = 0; i < 4; i++) {
			anntrain(test, inputs[i], &outputs[i]);

			if (counter++ % 1000 == 1) {
				error = 0;
			
				for (j = 0; j < 4; j++) {
					results = annrun(test, inputs[i]);
					error += pow(results[0] - outputs[i], 2.0);
					free(results);
				}
				print("error: %f\n", error);
			}
		}
	}
	print("error: %f, done\n", error);
}
