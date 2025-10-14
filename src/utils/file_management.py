from datetime import datetime
import os
import numpy as np
from pathlib import Path
from typing import Union, Literal

def file_title(title: str, dtype_suffix=".svg", short=False):
    '''
    Creates a file title containing the current time and a data-type suffix.

    Parameters
    ----------
    title: string
            File title to be used
    dtype_suffix: (default is ".svg") string
            Suffix determining the file type.
    short: boolean (default is False)
            If True, doesn't add h_min_sec to title (only Y-m-d).
    Returns
    -------
    file_title: string
            String to be used as the file title.
    '''
    if short:
        return datetime.now().strftime('%Y%m%d') + " " + title + dtype_suffix
    else:
        return datetime.now().strftime('%Y-%m-%d %H_%M_%S') + " " + title + dtype_suffix

def most_recent_file(directory: Union[Path, str], suffix_to_consider: str = None, file_title_keywords: [str] = None,
                     search_by: Literal["file-title", "meta-data"]= "file-title") -> str:
    """ search_by='file-title' works only with file-titles starting with YYYY-MM-DD HH_MM_SS (as created by the file_title method above) """
    if "." not in str(directory).split('/')[-1]:

        if search_by == "file-title":  # return file with most recent date in file title
            file_array, date_array = np.array([]), np.array([])
            for file in os.listdir(directory):
                # check for latest csv with ticker in title
                if suffix_to_consider is not None:
                    if not file.endswith(suffix_to_consider): continue
                elif '.DS_Store' in file: continue

                # check provided keywords
                if file_title_keywords is not None:  # if provided
                    if isinstance(file_title_keywords, str): file_title_keywords = [file_title_keywords]  # convert to list if required
                    match = True  # bool to only remain true if all keywords found
                    for file_title_keyword in file_title_keywords:
                        if file_title_keyword not in file: match = False
                    if not match: continue  # view next file

                din_datestring = file[:10]
                din_timestring = file[11:19].replace('_', ':')
                date = datetime.fromisoformat(din_datestring + ' ' + din_timestring)
                date_array = np.append(date_array, date)
                file_array = np.append(file_array, file)

            try:
                return directory / file_array[date_array.argsort()[-1]]
            except IndexError:
                raise ValueError("Provided directory doesn't contain files matching the provided criteria!")

        elif search_by == "meta-data":  # return file last edited
            # scan by meta-data:
            most_recent_file = None
            most_recent_time = 0

            for entry in os.scandir(directory):
                if entry.is_file():
                    # check for latest csv with ticker in title
                    if suffix_to_consider is not None:
                        if not entry.name.endswith(suffix_to_consider): continue

                    # check provided keywords
                    if file_title_keywords is not None:  # if provided
                        if isinstance(file_title_keywords, str): file_title_keywords = [
                            file_title_keywords]  # convert to list if required
                        match = True  # bool to only remain true if all keywords found
                        for file_title_keyword in file_title_keywords:
                            if file_title_keyword not in entry.name: match = False
                        if not match: continue  # view next file

                    # scan meta-data
                    mod_time = entry.stat().st_mtime  # modification time in seconds since epoch
                    if mod_time > most_recent_time:
                        most_recent_time = mod_time
                        most_recent_file = entry.name

            try:
                return directory / most_recent_file
            except TypeError:
                raise ValueError("Provided directory doesn't contain files matching the provided criteria!")
    else:
        raise NotADirectoryError("Provided path is not a directory (i.e. contains dots)!")


class TxtConfig:
    def __init__(self,
                 txt_file_path: Union[Path, str],):
        self.txt_file_path = txt_file_path

    @property
    def settings_dict(self) -> dict:
        temp_dict = {}
        with open(self.txt_file_path, "r") as file:
            for line in file:
                line.strip()

                # read entry and sanity check:
                entry = line.split(' --- ')
                if len(entry) != 2:
                    raise ValueError("Provided file may only contain lines structured as 'PROPERTY_NAME --- ENTRY'!")

                # save entry:
                temp_dict[entry[0].strip()] = entry[1].strip()
        return temp_dict

    def change_entry(self, entry, new_entry):
        temp_dict = self.settings_dict
        temp_dict[entry] = new_entry
        self._set_dict_to_file(temp_dict)

    def _set_dict_to_file(self, new_dict):
        # overwrite txt file with current dictionary content
        with open(self.txt_file_path, "w") as file:
            for key, value in new_dict.items():
                if isinstance(value, list):
                    value = ", ".join([str(e) for e in value])  # list formatting (can be retrieved via get_as_type)
                file.write(f"{str(key)} --- {str(value)}\n")

    def get_as_type(self, key, value_type: Literal["int", "float", "float_list", "str_list", "list", "bool", "str"]):
        value = self.settings_dict[key]  # retrieve value

        # format and return value:
        if value_type == "int": return int(value)
        elif value_type == "float": return float(value)
        elif value_type == "bool": return bool(value)
        elif value_type == "str": return str(value)
        elif value_type == "float_list":
            entries = value.split(', ')
            if len(entries) <= 1: raise ValueError("List entries need to be formatted as 'ENTRY_1, ENTRY_2, ENTRY_3, ...'!")
            return [float(e) for e in entries]
        elif value_type == "str_list" or value_type == "list":
            entries = value.split(', ')
            if len(entries) <= 1: raise ValueError("List entries need to be formatted as 'ENTRY_1, ENTRY_2, ENTRY_3, ...'!")
            return entries
        else:
            raise ValueError(f"Provided value type '{value_type}' is not recognized!")