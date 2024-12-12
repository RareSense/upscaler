from pydantic import BaseModel, validator
from typing import Dict

from pydantic import BaseModel, Field, validator
from typing import Dict, Optional

class UpscaleRequest(BaseModel):
    input: Dict[str, str]  # Expecting a dictionary with a "target_image" key
    upscaling_factor: Optional[int] = Field(2, description="The upscaling factor, ranging from 2 to 5.")

    @validator("input")
    def validate_input_key(cls, value):
        if "target_image" not in value:
            raise ValueError("The 'input' dictionary must contain a 'target_image' key.")
        return value

    @validator("upscaling_factor")
    def validate_upscaling_factor(cls, value):
        if value < 2 or value > 5:
            raise ValueError("The 'upscaling_factor' must be between 2 and 5.")
        return value


class UpscaleResponse(BaseModel):
    output: Dict[str, str]  # The dictionary contains an "image" key
