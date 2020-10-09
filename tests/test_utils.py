import pytest

from plkit.utils import *

@pytest.mark.parametrize('tvt_ratio,expected', [
    (None, None),
    (.7, (.7, None, None)),
    ((.7, .1), (.7, [.1], None)),
    ((.7, .15, .15), (.7, [.15], [.15])),
])
def test_normalize_tvt_ratio(tvt_ratio, expected):
    assert normalize_tvt_ratio(tvt_ratio) == expected

def test_check_config():
    with pytest.raises(PlkitConfigException):
        check_config({}, 'batch_size')
