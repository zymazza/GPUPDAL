(filters.radiusassign)=

# filters.radiusassign

The **radius assign filter** allows you update the value of a dimension (using
an assignment expression) for specific points depending on their neighbors
in a given radius:
For each point in the domain src_domain_, if it has any neighbor with a
distance lower than radius_ that belongs to the domain reference_domain_,
it is updated using the expression update_expression_.

```{eval-rst}
.. embed::
```

## Example

This pipeline updates the Keypoint dimension of all points with classification
1 to 2 (unclassified and ground) that are closer than 1 meter from a point with
classification 6 (building)


```json

  [
      "las/4_6.las",
      {
          "type" : "filters.radiusassign",
          "src_domain" : "Classification[1:2]",
          "reference_domain" : "Classification[6:6]"
          "radius" : 1
          "update_expression": "Keypoint = 1"
      },
      "output.las"
  ]

```

## Options

src_domain

: A {ref}`range <ranges>` which selects points to be processed by the filter.
  Can be specified multiple times.  Points satisfying any range will be
  processed

reference_domain

: A {ref}`range <ranges>` which selects points that can are considered as
  potential neighbors. Can be specified multiple times.

radius

: A positive float which specifies the radius for the neighbors search.

update_expression

: A list of {ref}`assignment expressions <Assignment Expressions>` to be applied to
  the points that satisfy the radius search.   The list of values is evaluated in order.

is3d

: Use three-dimensional distance. The default is false, which uses XY distance
  and may apply the vertical caps below.

max2d_above

: In two-dimensional mode, reject a reference point more than this distance
  above the source point. A negative value disables the cap. Equality passes.

max2d_below

: In two-dimensional mode, reject a reference point more than this distance
  below the source point. A negative value disables the cap. Equality passes.

## PDG execution

The default compatibility path preserves the pinned PDAL implementation,
including strict `distance < radius`, self-inclusion, OR semantics across each
domain list, the source-order selection prepass, ordered assignment evaluation,
diagnostics, and output order.

For compatible linear pipelines, `pdg resident` can build one planner-owned
2D or 3D shared radius index and evaluate the source/reference membership query
on CUDA. Exact compatibility mode then evaluates the original PDAL assignment
expressions in their original order on the host and re-uploads only written
columns needed by a later resident stage. Coordinate-writing expressions,
unsupported options or graph shapes, mixed radius/kNN regions, and regions
mixing 2D and 3D requests retain the exact host path.

The SM 89 placement profile selects this resident executor only inside its
measured 250,000–21,970,934-point envelope. On the reference RTX 4090, the
21,970,934-point format-7 B2 lane is byte-identical and measured 5.503x faster
than pinned PDAL (44.461 s versus 8.080 s median). This is a resident hybrid
stage—CUDA performs the radius selection while the assignment-expression
finale remains on the host—not a wholly device-arithmetic implementation. See
`BENCHMARKS.md` B0019 for the complete protocol and report hashes.


```{include} filter_opts.md
```

