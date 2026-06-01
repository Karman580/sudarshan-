import re

class ChecksumValidator:
    def __init__(self):
        # Verhoeff tables
        self.d = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
            [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
            [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
            [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
            [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
            [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
            [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
            [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        ]
        self.p = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
            [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
            [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
            [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
            [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
            [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
            [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
        ]
        self.inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]

    def extract_and_validate(self, text):
        """Finds all 12 digit combinations in text and runs Verhoeff"""
        # Find 12 digits, ignoring spaces/hyphens
        clean_text = re.sub(r'[^0-9]', '', text)
        found_uids = []
        for i in range(len(clean_text) - 11):
            subset = clean_text[i:i+12]
            if len(subset) == 12:
                is_valid = self._verhoeff_validate(subset)
                if is_valid:
                    found_uids.append(subset)
                    
        return {
            "uids_found": len(found_uids),
            "uids": found_uids,
            "valid": len(found_uids) > 0
        }

    def _verhoeff_validate(self, uid):
        c = 0
        inverted_uid = uid[::-1]
        for i, val in enumerate(inverted_uid):
            c = self.d[c][self.p[i % 8][int(val)]]
        return c == 0
