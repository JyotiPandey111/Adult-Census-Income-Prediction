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
    
    
    
    