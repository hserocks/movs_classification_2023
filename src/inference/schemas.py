from datetime import datetime
from pydantic import BaseModel, validator, constr, Field
import pytz
from dateutil import parser # python-dateutil



# class InferenceCreate(BaseModel):
#     model: str = 'resnet'
#     link: str = 'https://cdn.britannica.com/71/234471-050-093F4211/shiba-inu-dog-in-the-snow.jpg'
    
    


allowed_models = ['resnet', 'vit', 'resnet_svm', 'vit_svm', 'resnet_xgb', 'vit_xgb'] 

class InferenceCreate(BaseModel):
    model: constr(strict=True) = Field(
        ...,
        description="The model type",
        example="resnet",
        pattern='^(' + '|'.join(allowed_models) + ')$'
    )
    link: str = Field(
        default='https://cdn.britannica.com/71/234471-050-093F4211/shiba-inu-dog-in-the-snow.jpg',
        description="The link to the image",
        example="https://cdn.britannica.com/71/234471-050-093F4211/shiba-inu-dog-in-the-snow.jpg"
    )
    
    class Config:
        from_attributes=True
        populate_by_name  = True


    # Custom validator for the 'model' field
    @validator('model')
    def model_must_be_in_allowed_list(cls, v):
        if v not in allowed_models:
            raise ValueError(f'model must be one of {allowed_models}')
        return v


class InferenceGimages(BaseModel):
    model: constr(strict=True) = Field(
        ...,
        description="The model type",
        example="resnet",
        pattern='^(' + '|'.join(allowed_models) + ')$'
    )
    query: str = Field(
        default='Cat',
        description="Query for Google Images",
        example="Cat"
    )
    
    class Config:
        from_attributes=True
        populate_by_name  = True


    # Custom validator for the 'model' field
    @validator('model')
    def model_must_be_in_allowed_list(cls, v):
        if v not in allowed_models:
            raise ValueError(f'model must be one of {allowed_models}')
        return v
    
    # id: int
    # date: datetime = datetime.now().astimezone(pytz.utc).replace(tzinfo=None)
    # type: str

# class InferenceCreate(BaseModel):
#     id: int
#     model: str
#     link: str
#     date: datetime
#     type: str

    # @validator('date', pre=True)
    # def ensure_datetime_tz_aware(cls, v):
    #     """Ensure the 'date' field is a timezone-aware datetime."""
    #     if isinstance(v, str):
    #         # Attempt to parse the string into a datetime object
    #         try:
    #             v = parser.parse(v)
    #         except ValueError:
    #             raise ValueError("Unable to parse the date string")

    #     if isinstance(v, datetime):
    #         # If the 'date' is a datetime but not timezone-aware, make it aware
    #         if v.tzinfo is None:
    #             return v.replace(tzinfo=pytz.UTC)
    #         return v
    #     else:
    #         raise ValueError("Invalid date format")
    #     return v