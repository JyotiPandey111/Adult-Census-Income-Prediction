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

class education(str, Enum):
    SeventhEightthGrade = ' 7th-8th'
    NinthGrade = ' 9th'
    TenthGrade = ' 10th'
    EleventhGrade = ' 11th'
    TwelveGrade = ' 12th'
    HighSchoolGrad = ' HS-grad'
    Bachelor = ' Bachelors'
    SomeCollege = ' Some-college'
    Masters = ' Masters'
    AssociateVOC = ' Assoc-voc'
    AssociateAcademic = ' Assoc-acdm'
    ProfessorSchool = ' Prof-school'
    Doctorate = ' Doctorate'
    #Preschool =  'Rare-var'
    #FirstToForthGrade = 'Rare-var'
    #FifthToSixthGrade = 'Rare-var'


class workclass(str,Enum):
   private = ' Private'
   selfemp = ' Self-emp-not-inc'
   local = ' Local-gov'
   stategov = ' State-gov'
   selfinc = ' Self-emp-inc'
   federal = ' Federal-gov'
   nopay = ' Without-pay'
   nowork = ' Never-worked'



class maritalstatus(str,Enum):
    never = ' Never-married'
    civ =  ' Married-civ-spouse'
    widowed =  ' Widowed'
    div =  ' Divorced'
    sep =  ' Separated'
    mar =  ' Married-spouse-absent'
    AF =  ' Married-AF-spouse'

class occupation(str,Enum): 
   other = ' Other-service'
   nachine = ' Machine-op-inspct'
   spec = ' Prof-specialty'
   craft = ' Craft-repair'
   sales = ' Sales'
   transport = ' Transport-moving'
   cler = ' Adm-clerical'
   handler = ' Handlers-cleaners'
   manag = ' Exec-managerial'
   tech = ' Tech-support'
   prot = ' Protective-serv'
   farm = ' Farming-fishing'
   priv = ' Priv-house-serv'
   arm = ' Armed-Forces'


class race(str,Enum): 
   black = ' Black'
   white = ' White'
   asia = ' Asian-Pac-Islander'
   amer = ' Amer-Indian-Eskimo'
   other = ' Other'

class sex(str,Enum):
    fe = ' Female'
    ma = ' Male'
    
    
    
    