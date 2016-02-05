from __future__ import print_function, division, unicode_literals
from codecs import open
import sys


def cut_into_small_chunks(seq, len_chunk, len_overlap):
    if len(seq) < len_chunk:
        return [seq]
    else:
        return [seq[:len_chunk]] + cut_into_small_chunks(seq[len_chunk - len_overlap:], len_chunk, len_overlap)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Call with arguments CHUNKSIZE and OVERLAP_LENGTH', file=sys.stderr)
        sys.exit(1)
    chunksize = int(sys.argv[1])
    overlap = int(sys.argv[2])
    for words in (line.strip('\n').split() for line in sys.stdin):
        for new_line in cut_into_small_chunks(words, chunksize, overlap):
            print(' '.join(new_line))