.. _custom_objectives:

=================
Custom Objectives
=================

.. contents:: Table of Contents


How do you write your own objective function to optimize?

The Objective Class
===================

In Lucent, an objective is an object with a ``description``, a ``name`` and an ``objective_func``. It has a call method which takes a single argument
(the model) and evaluates ``objective_func`` on that input. 

Objectives can be summed, subtracted, multiplied and divided. 

.. admonition:: Summation

   If you want to use ``sum`` on an iterable of Objectives, then you should use the classmethod ``Objective.sum(my_iterable)`` instead of ``sum(my_iterable)``.
   Otherwise, the resulting Objective instance will have a nested description like ``Sum(obj_1 + Sum(obj_2 + Sum(...)))``.

In principle, any possible differentiable objective function could be used to construct an Objective instance.

For the full documentation see :doc:`` TODO


The ``@wrap_objective()`` wrapper
=================================

....

