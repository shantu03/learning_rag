from langchain_huggingface import HuggingFaceEmbeddings
import torch



def _get_device():
    """ AVAIABLE HARDWARE -> cuda/cpu
    """
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    print(f"using device {device.upper() } .!")

    return device

DEVICE=_get_device()    

def get_MiniLM():
    """
    STRENGTHS: Lightning-fast, ultra-lightweight (~90 MB), low memory usage.
    BEST FOR: Rapid testing, prototyping, and CPU-heavy learning environments.
    """


    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )

def get_mpnet():
    """
    STRENGTHS: Highly accurate English search, excellent semantic grasp (~420 MB).
    BEST FOR: Standard English RAG systems where accuracy matters more than speed."""
    return  HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": DEVICE}
    )
def get_bge_m3() -> HuggingFaceEmbeddings:
    """
    STRENGTHS: Flagship multilingual capabilities, supports multiple languages (~2.2 GB).
    BEST FOR: Mixed-language datasets, non-English search, and global applications.
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": DEVICE}
    )

def get_nomic() -> HuggingFaceEmbeddings:
    """
    STRENGTHS: Enormous context window (8,192 tokens), great for long text (~280 MB).
    BEST FOR: Embedding full pages, large legal documents, or entire books at once.
    """
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"device": DEVICE}
    )