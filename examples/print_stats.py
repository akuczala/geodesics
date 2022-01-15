import pstats
from pstats import SortKey
p = pstats.Stats('examples/raytace-stats')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()