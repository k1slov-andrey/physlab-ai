from core.lab_registry import list_labs
from core.recommendation_engine import build_feedback
from core.schemas import ModelPrediction

def main():
    for lab in list_labs(): print(f"{lab.lab_id}: {lab.short_title} [{lab.implementation_status}]")
    p=ModelPrediction("boyle_mariotte","air_leak",0.91,{"air_leak":0.91},{"pv_slope":-0.012})
    print(build_feedback(p))
if __name__=="__main__": main()
