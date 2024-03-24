from datetime import datetime
from pydantic import BaseModel, validator, constr, Field
import pytz
from dateutil import parser # python-dateutil



# class InferenceCreate(BaseModel):
#     model: str = 'resnet'
#     link: str = 'https://cdn.britannica.com/71/234471-050-093F4211/shiba-inu-dog-in-the-snow.jpg'
    
    


allowed_models = ['resnet', 'vit', 'resnet_svm', 'vit_svm', 'resnet_xgb', 'vit_xgb'] 
allowed_features = ['resnet', 'vit']
allowed_fmodels = ['svm', 'xgb']

class ModelEvaluate(BaseModel):
    model: constr(strict=True) = Field(
        ...,
        description="The model type",
        example="resnet",
        pattern='^(' + '|'.join(allowed_models) + ')$'
    )
    data: str = Field(
        default='Data_small',
        description="Data source (folder with images)",
        example='Data_small'
    )
    evaluate_only: bool = Field(
        default=True,
        description="Evaluate only (don't retrain the model)",
        example=True
    )
    
    # Custom validator for the 'model' field
    @validator('model')
    def model_must_be_in_allowed_list(cls, v):
        if v not in allowed_models:
            raise ValueError(f'model must be one of {allowed_models}')
        return v

class FeatureModelEvaluate(BaseModel):
    features: constr(strict=True) = Field(
        ...,
        description="Feature type",
        example="resnet",
        pattern='^(' + '|'.join(allowed_features) + ')$'
    )
    fmodel: constr(strict=True) = Field(
        ...,
        description="The model type",
        example="svm",
        pattern='^(' + '|'.join(allowed_fmodels) + ')$'
    )
    evaluate_only: bool = Field(
        default=True,
        description="Evaluate only (don't retrain the model)",
        example=True
    )


    @validator('features')
    def features_must_be_in_allowed_list(cls, v):
        if v not in allowed_features:
            raise ValueError(f'features must be one of {allowed_features}')
        return v
    
    
    @validator('fmodel')
    def model_must_be_in_allowed_list(cls, v):
        if v not in allowed_fmodels:
            raise ValueError(f'model must be one of {allowed_fmodels}')
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