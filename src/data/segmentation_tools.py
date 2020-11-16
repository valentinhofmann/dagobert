from pathlib import Path

# Initialize prefix list
prefix_list = []
with open(str(Path(__file__).parent.parent.parent / 'data/external/bert_prefixes.txt'), 'r') as file:
    for line in file:
        if line.strip() == '':
            continue
        prefix_list.append(line.strip().lower())

# Initialize suffix list
suffix_list = []
with open(str(Path(__file__).parent.parent.parent / 'data/external/bert_suffixes.txt'), 'r') as file:
    for line in file:
        if line.strip() == '':
            continue
        suffix_list.append(line.strip().lower())

# Initialize vocabulary core
core = set()
with open(str(Path(__file__).parent.parent.parent / 'data/external/bert_core.txt'), 'r') as file:
    for line in file:
        if line.strip() == '':
            continue
        core.add(line.strip().lower())


# Define function to detect vowels
def is_vowel(char):
    return char.lower() in 'aeiou'


# Define function to detect consonants
def is_cons(char):
    return char.lower() in 'bdgptkmnlrszfv'


# Define function to find segmentation of form in a rule-based way
def segment(form, control, prefixes=prefix_list, suffixes=suffix_list):

    found_prefixes = []
    prefix_happy = True

    # Outer loop to check prefixes
    while prefix_happy:

        found_suffixes = []
        form_temp = form
        suffix_happy = True

        # Inner loop to check suffixes
        while suffix_happy:

            if form_temp == '':
                break

            if form_temp in control:

                return found_prefixes[:], form_temp, found_suffixes[::-1]

            elif len(form_temp) > 2 and is_cons(form_temp[-1]) and len(found_suffixes) > 0 \
                    and is_vowel(found_suffix[0]) and form_temp + 'e' in control:

                form_temp = form_temp + 'e'

                return found_prefixes[:], form_temp, found_suffixes[::-1]

            elif len(form_temp) > 2 and is_cons(form_temp[-1]) and form_temp[-1] == form_temp[-2] \
                    and is_vowel(form_temp[-3]) and len(found_suffixes) > 0 and form_temp[:-1] in control:

                form_temp = form_temp[:-1]

                return found_prefixes[:], form_temp, found_suffixes[::-1]

            # Need to find suffix to stay in inner loop
            suffix_happy = False
            found_suffix = ''
            for suffix in suffixes:
                if form_temp.endswith(suffix):
                    suffix_happy = True
                    if suffix == 'ise':
                        found_suffix = 'ize'
                    else:
                        found_suffix = suffix
                    found_suffixes.append(found_suffix)
                    form_temp = form_temp[:-len(suffix)]
                    break

            # Check for special phonological alternations
            try:
                if found_suffix in {'ation', 'ate'} and form_temp[-4:] == 'ific':
                    form_temp = form_temp[:-4] + 'ify'
                elif found_suffix == 'ness' and form_temp[-1] == 'i':
                    form_temp = form_temp[:-1] + 'y'
                elif form_temp[-4:] == 'abil':
                    form_temp = form_temp[:-4] + 'able'
            except IndexError:
                continue

        # Need to find prefix to stay in outer loop
        prefix_happy = False
        for prefix in prefixes:
            if form.startswith(prefix):
                # Check addition of false prefixes
                prefix_happy = True
                if prefix in {'im', 'il', 'ir'}:
                    found_prefix = 'in'
                else:
                    found_prefix = prefix
                found_prefixes.append(found_prefix)
                form = form[len(prefix):]
                break

    return ''


# Define function to return segmentation of form
def derive(form, control=core, mode='bundles'):
    try:
        prefixes, root, suffixes = segment(form, control)
        if mode == 'roots':
            return root
        if mode == 'bundles':
            return ''.join(p + '_' for p in prefixes), root, ''.join('_' + s for s in suffixes)
        if mode == 'morphemes':
            return prefixes, root, suffixes
    except ValueError:
        if mode == 'roots':
            return form
        if mode == 'bundles':
            return '', form, ''
        if mode == 'morphemes':
            return [], form, []
