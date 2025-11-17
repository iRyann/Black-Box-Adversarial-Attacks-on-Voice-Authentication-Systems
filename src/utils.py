
def getSpeakerID(sample_name: str) -> str:
    parts = sample_name.split("_")
    return parts[1]

def getSampleIndex(sample_name:str) -> str:
    parts = sample_name.split("_")
    return parts[2][:-4]
