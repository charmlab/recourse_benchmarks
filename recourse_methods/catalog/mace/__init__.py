# flake8: noqa
import os
import sys

from data.catalog.loadData import *
from models.catalog.loadModel import *

from .model import MACE

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
