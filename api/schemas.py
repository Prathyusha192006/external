from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator

class CustomerInput(BaseModel):
    customerID: Optional[str] = Field("CUST-9999", description="Unique Customer Identifier")
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender ('Male' or 'Female')")
    SeniorCitizen: Literal[0, 1] = Field(..., description="Whether customer is senior citizen (0 or 1)")
    Partner: Literal["Yes", "No"] = Field(..., description="Whether customer has partner ('Yes' or 'No')")
    Dependents: Literal["Yes", "No"] = Field(..., description="Whether customer has dependents ('Yes' or 'No')")
    tenure: int = Field(..., ge=0, le=120, description="Tenure in months (0 to 120)")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Phone service subscription ('Yes' or 'No')")
    MultipleLines: Literal["No phone service", "No", "Yes"] = Field(..., description="Multiple lines status")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Internet service type")
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(..., description="Online security add-on")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., description="Online backup add-on")
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(..., description="Device protection add-on")
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(..., description="Tech support add-on")
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(..., description="Streaming TV add-on")
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(..., description="Streaming movies add-on")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract duration")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Paperless billing ('Yes' or 'No')")
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly charges ($)")
    TotalCharges: Union[float, str] = Field(..., description="Total cumulative charges ($) or string")

    @field_validator("TotalCharges")
    def validate_total_charges(cls, v):
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("TotalCharges cannot be negative.")
        elif isinstance(v, str):
            v_clean = v.strip()
            if v_clean != "":
                try:
                    val = float(v_clean)
                    if val < 0:
                        raise ValueError("TotalCharges cannot be negative.")
                except ValueError:
                    raise ValueError("TotalCharges must be a valid number or empty string.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }
    }

class PredictionResponse(BaseModel):
    customerID: str
    churn_prediction: str
    churn_class: int
    churn_probability: float
    risk_level: str
    risk_factors: List[str]
    engineered_features: Dict[str, float]
    model_version: str

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerInput]

class MetricsResponse(BaseModel):
    model_name: str
    model_stage: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    hyperparameters: Dict[str, Any]
    last_evaluated: Optional[str] = None
