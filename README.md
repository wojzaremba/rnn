RNN training
============
Call ./main.py model, we support currently models "mock", and "pennchr".


Sequence generation
===================
Call ./gen.py model text (e.g. ./gen pennchr My_name_is). It generates sequences conditioned on the previous data.

Setting up hooks
================
    If you would like to submit anything, please setup testing hooks on your machine.
    cd .git/hooks
    .git/hooks$ ln -s ../../PRESUBMIT.py pre-commit
    chmod a+x ./pre-commit
