from monai.transforms.transform import MapTransform
from monai.config.type_definitions import KeysCollection
from typing import Union, Dict, Any
from pathlib import Path
from .text_io import load_tokens_for_id


class AddTextTokensd(MapTransform):
    """
    MapTransform that reads "ID" from the sample dict, loads MS-RAW tokens+mask
    from text_root, and inserts:
      - d["TXT"]      = (L,Ct) float32 tensor
      - d["TXT_MASK"] = (L,)   bool tensor
    """

    def __init__(
        self,
        keys: KeysCollection,
        text_root: Union[str, Path],
        allow_missing: bool = False,
    ):
        super().__init__(keys)
        self.text_root = Path(text_root)
        self.allow_missing = allow_missing

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        # Expect an "ID" string or the key passed as 'keys'
        if isinstance(self.keys, (list, tuple)) and len(self.keys) == 1:
            id_key = self.keys[0]
        else:
            id_key = self.keys if isinstance(self.keys, str) else "ID"
        rid = d.get(id_key)
        if rid is None:
            if self.allow_missing:
                return d
            raise KeyError(f"Sample is missing '{id_key}' needed to load text tokens.")

        try:
            tokens, mask = load_tokens_for_id(self.text_root, str(rid))
            d["TXT"] = tokens  # (L, Ct)
            d["TXT_MASK"] = mask  # (L,)
        except FileNotFoundError:
            if self.allow_missing:
                return d
            raise
        return d
