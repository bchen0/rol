import glob
import os
import sys


def normalize_relative_include_target(target):
    """
    Rewrite only truly relative includes.

    Examples:
      ../../TOOLS/dynpde.hpp -> dynpde.hpp
      ../foo/bar.hpp         -> bar.hpp
      ./baz.hpp              -> baz.hpp

    This is intentionally conservative: it only applies to includes
    starting with ./ or ../. Logical include paths such as
    desul/atomics/Atomic_Ref.hpp are preserved.
    """
    if target in [
        'storage_class.h',
        'cuda_cc7_asm_atomic_op.inc_predicate',
        'cuda_cc7_asm_atomic_fetch_op.inc_predicate'
    ]:
        return target

    norm = os.path.normpath(target)
    return os.path.basename(norm)


def rewrite_include_line(line):
    """
    Rewrite include lines conservatively:

    - Leave angle-bracket includes unchanged.
    - Leave normal quoted includes unchanged.
    - Only rewrite quoted includes that start with ./ or ../
      into angle-bracket includes with a normalized target.
    """
    stripped = line.lstrip()
    if not stripped.startswith('#include'):
        return line

    # If this line already uses angle brackets, leave it alone.
    if '<' in stripped and '>' in stripped:
        return line

    first_quote = line.find('"')
    second_quote = line.find('"', first_quote + 1)

    # If it's not a quoted include, leave it alone.
    if first_quote == -1 or second_quote == -1:
        return line

    target = line[first_quote + 1:second_quote]

    # Only rewrite truly relative includes.
    if target.startswith('../') or target.startswith('./'):
        new_target = normalize_relative_include_target(target)
        return line[:first_quote] + '<' + new_target + '>' + line[second_quote + 1:]

    # Preserve ordinary quoted includes, including sibling/local includes
    # like "default_accessor.hpp" and rooted logical includes like
    # "desul/atomics/Atomic_Ref.hpp".
    return line


def make_all_includes(all_include_filename, folders):
    all_includes = []
    for folder in folders:
        for filename in (
            glob.glob(f'{folder}/**/*.hpp', recursive=True) +
            glob.glob(f'{folder}/**/*.cpp', recursive=True) +
            glob.glob(f'{folder}/**/*.h', recursive=True) +
            glob.glob(f'{folder}/**/*.cc', recursive=True) +
            glob.glob(f'{folder}/**/*.c', recursive=True)
        ):
            with open(filename, 'r') as fh:
                for line in fh:
                    if line.lstrip().startswith('#include'):
                        all_includes.append(rewrite_include_line(line).strip())
    all_includes = list(set(all_includes))
    all_includes.sort()
    with open(all_include_filename, 'w') as fh:
        for include in all_includes:
            fh.write(f'{include}\n')
    return all_include_filename


def make_all_includes_from_filenames(all_include_filename, filenames):
    all_includes = []
    for filename in filenames:
        with open(filename, 'r') as fh:
            for line in fh:
                if line.lstrip().startswith('#include'):
                    all_includes.append(rewrite_include_line(line).strip())
    all_includes = list(set(all_includes))
    all_includes.sort()
    with open(all_include_filename, 'w') as fh:
        for include in all_includes:
            fh.write(f'{include}\n')
    return all_include_filename


# https://github.com/RosettaCommons/binder/issues/212
def copy_and_rewrite_includes(filenames, filenames_without_dir, to_dir):
    """
    Copy headers into the temporary include tree and rewrite only the
    include directives that truly need rewriting.
    """
    for i in range(len(filenames)):
        filename = filenames[i]
        filename_without_dir = filenames_without_dir[i]
        path = os.path.dirname(filename_without_dir)

        out_dir = os.path.join(to_dir, path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        try:
            with open(filename, 'r') as from_f:
                lines = from_f.readlines()
        except UnicodeDecodeError:
            with open(filename, 'r', encoding='iso-8859-1') as from_f:
                lines = from_f.readlines()
        except PermissionError:
            continue

        out_file = os.path.join(to_dir, filename_without_dir)
        with open(out_file, 'w') as to_f:
            for line in lines:
                if line.lstrip().startswith('#include'):
                    line = rewrite_include_line(line)
                to_f.write(line)


if __name__ == '__main__':
    CMAKE_CURRENT_SOURCE_DIR = sys.argv[1]
    CMAKE_CURRENT_BINARY_DIR = sys.argv[2]
    all_header_list_with_dir = sys.argv[3]
    all_header_list_without_dir = sys.argv[4]
    binder_include_name = sys.argv[5]

    with open(all_header_list_with_dir, 'r') as fh:
        all_include_filenames_with_dir = fh.read().splitlines()

    with open(all_header_list_without_dir, 'r') as fh:
        all_include_filenames_without_dir = fh.read().splitlines()

    copy_and_rewrite_includes(
        all_include_filenames_with_dir,
        all_include_filenames_without_dir,
        CMAKE_CURRENT_BINARY_DIR + '/include_tmp'
    )

    make_all_includes_from_filenames(
        binder_include_name,
        [CMAKE_CURRENT_SOURCE_DIR + '/src/Binder_Input.hpp']
    )
