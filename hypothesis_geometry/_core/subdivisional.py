from collections.abc import Sequence
from typing import Generic

from ground.context import Context
from ground.enums import Orientation
from ground.hints import Point
from reprit.base import generate_repr
from typing_extensions import Self

from .hints import HasCustomRepr, ScalarT


class QuadEdge(HasCustomRepr, Generic[ScalarT]):
    """
    Based on:
        quad-edge data structure.

    Reference:
        https://en.wikipedia.org/wiki/Quad-edge
        http://www.sccg.sk/~samuelcik/dgs/quad_edge.pdf
    """

    @classmethod
    def from_endpoints(
        cls,
        start: Point[ScalarT],
        end: Point[ScalarT],
        /,
        *,
        context: Context[ScalarT],
    ) -> Self:
        """Creates new edge from endpoints."""
        result, opposite = (
            cls(start, context=context),
            cls(end, context=context),
        )
        rotated, triple_rotated = cls(context=context), cls(context=context)
        result._left_from_start = result
        opposite._left_from_start = opposite
        rotated._left_from_start = triple_rotated
        triple_rotated._left_from_start = rotated
        result._rotated = rotated
        rotated._rotated = opposite
        opposite._rotated = triple_rotated
        triple_rotated._rotated = result
        return result

    @property
    def end(self, /) -> Point[ScalarT]:
        """
        aka "Dest" in L. Guibas and J. Stolfi notation.
        """
        return self.opposite.start

    @property
    def left_from_end(self, /) -> Self:
        """
        aka "Lnext" in L. Guibas and J. Stolfi notation.
        """
        return self.rotated.opposite.left_from_start.rotated

    @property
    def left_from_start(self, /) -> Self:
        """
        aka "Onext" in L. Guibas and J. Stolfi notation.
        """
        result = self._left_from_start
        assert result is not None
        return result

    @property
    def opposite(self, /) -> Self:
        """
        aka "Sym" in L. Guibas and J. Stolfi notation.
        """
        return self.rotated.rotated

    @property
    def right_from_end(self, /) -> Self:
        """
        aka "Rprev" in L. Guibas and J. Stolfi notation.
        """
        return self.opposite.left_from_start

    @property
    def right_from_start(self, /) -> Self:
        """
        aka "Oprev" in L. Guibas and J. Stolfi notation.
        """
        return self.rotated.left_from_start.rotated

    @property
    def rotated(self, /) -> Self:
        """
        aka "Rot" in L. Guibas and J. Stolfi notation.
        """
        result = self._rotated
        assert result is not None
        return result

    @property
    def start(self, /) -> Point[ScalarT]:
        """
        aka "Org" in L. Guibas and J. Stolfi notation.
        """
        result = self._start
        assert result is not None
        return result

    __slots__ = '_left_from_start', '_rotated', '_start', 'context'

    def __init__(
        self,
        start: Point[ScalarT] | None = None,
        left_from_start: Self | None = None,
        rotated: Self | None = None,
        /,
        *,
        context: Context[ScalarT],
    ) -> None:
        (self.context, self._left_from_start, self._rotated, self._start) = (
            context,
            left_from_start,
            rotated,
            start,
        )

    __repr__ = generate_repr(from_endpoints)

    def connect(self, other: Self, /) -> Self:
        """Connects the edge with the other."""
        result = self.from_endpoints(
            self.end, other.start, context=self.context
        )
        result.splice(self.left_from_end)
        result.opposite.splice(other)
        return result

    def delete(self, /) -> None:
        """Deletes the edge."""
        self.splice(self.right_from_start)
        self.opposite.splice(self.opposite.right_from_start)

    def orientation_of(self, point: Point[ScalarT], /) -> Orientation:
        """Returns orientation of the point relative to the edge."""
        return self.context.angle_orientation(self.start, self.end, point)

    def splice(self, other: Self, /) -> None:
        """Splices the edge with the other."""
        alpha = self.left_from_start.rotated
        beta = other.left_from_start.rotated
        self._left_from_start, other._left_from_start = (  # noqa: SLF001
            other.left_from_start,
            self.left_from_start,
        )
        alpha._left_from_start, beta._left_from_start = (  # noqa: SLF001
            beta.left_from_start,
            alpha.left_from_start,
        )

    def flip(self, /) -> None:
        """Flips diagonal of a quadrilateral."""
        side = self.right_from_start
        opposite = self.opposite
        opposite_side = opposite.right_from_start
        self.splice(side)
        opposite.splice(opposite_side)
        self.splice(side.left_from_end)
        opposite.splice(opposite_side.left_from_end)
        self._start = side.end
        opposite._start = opposite_side.end  # noqa: SLF001


def to_edge_neighbours(
    edge: QuadEdge[ScalarT], /
) -> Sequence[QuadEdge[ScalarT]]:
    candidate = edge.left_from_start
    return (
        (candidate, candidate.right_from_end)
        if (edge.orientation_of(candidate.end) is Orientation.COUNTERCLOCKWISE)
        else ()
    )
