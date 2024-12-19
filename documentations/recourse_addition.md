# RECOURSE METHOD ADDITION IN THE CHARM LAB RECOURSE LIBRARY

This document serves as a guideline that itemizes the steps required in adding a new recourse method in the CHARM Lab recourse library. The steps itemized in this document are intentionally designed to be general, and applicable to any recourse method algorithm.

1. Since this recourse library is inspired and built on the CARLA library, the first step would be to extensively read and understand the associated interfaces and classes associated with the recourse methods class as defined in the CARLA documentation:

   - [Welcome to CARLA’s documentation](https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/)
   - [Recourse Methods](https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/recourse.html)
   - [How to use CARLA](https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/notebooks/how_to_use_carla.html)
   - [Benchmarking](https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/benchmarking.html)
   - [Examples](https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/examples.html)

   By extensively reading the links above, a high level intuition on how carla, its recourse models, and benchmarking tool operates should be derived. This should help in shaping your perspective on how to effetively contribute to this repository.

2. For any recourse method to be added to this repository, there should be a counterfactual generation algorithm inherent to the method. If the recourse method intended to be included is available in the PYPI repository, then it may be added to the `requirements-dev.txt` file, as well as the `install_requires` list in the `setup.py` file.

3. Conversely, if the algorithm does not exist as a published python package, then the counterfactual generation algorithm would be implemented, and stored locally in the repository.

4. New recourse methods are added as a folder in the methods/catalog directory.

5. New additions come with customary `__init__.py`, `model.py` and `reproduce.py` files. The `__init__.py` file is simply an initialization file that allows the created recourse method class from `model.py` to be called on. For instance, here is the [dice](https://github.com/charmlab/recourse_benchmarks/tree/main/methods/catalog/dice) recourse method, added to the repository with these files defined.

6. The `model.py` file is where the recourse method is explicitly defined, according to the `RecourseMethod` interface. Example usage is seen [here](https://github.com/charmlab/recourse_benchmarks/blob/main/methods/catalog/dice/model.py). As indicated above, implementation may differ if the algorithm/method exists as a python package in PYPI or otherwise.

   - If the former is the case:

     - The dice recourse method is the primal example of how to add such methods. The class (named after the method), has default methods: `__init__` and `get_counterfactuals`. The former is primarily to initialize class attributes and parent classes. The `get_counterfactuals` method is where the algorithm is called to generated counterfactual results.

   - If the later is the case:

     - Recourse methods whose algorithmic implementation do not exist as a published python package, may need to create a `/library` folder in the root directory of the associated recourse method that exists in the `methods/catalog `directory. For example: here’s the [library folder](https://github.com/charmlab/recourse_benchmarks/tree/main/methods/catalog/wachter/library) created for the watcher recourse method, to house its algorithmic implementation.

     - As highlighted in the previous paragraph, the library folder is intended to house the algorithmic implementation of the recourse method.Consequently, all associated implementations should be housed in this folder. However, there are exceptions where the algorithms are non-complex, and do not require a separation of files. If the implemented algorithm is short, and can be fit within the `model.py` file, then there is no need to create a separate library folder.
     - The final counterfactual generating function should be referenced in the `model.py file`, and specifically in the `get_counterfactuals` function.

7. The `reproduce.py` file is the file that contains unit tests that replicate the experiments presented in the corresponding research paper, ensuring that the results obtained are consistent with those reported in the paper, within an acceptable margin of error. You should ensure the added unit tests in this file run and pass successfully.

8. Following the successful addition of the recourse method to the repository, the new method and its hyperparameters may be appended to the `experiment_setup.yaml` file.

9. Finally, the newly added method may be included and instantiated in the `run_experiment.py` file, and specifically in the `initialize_recourse_methods` function.
