from pydantic import BaseModel
from enum import Enum

#class IncomePrediction(BaseModel):
class relationship(str, Enum):
    Unmarried = ' Unmarried'
    Husband = ' Husband'
    NotInFamily = ' Not-in-family'
    OwnChild=' Own-child'
    Wife = ' Wife'
    OtherRelative = ' Other-relative'

    