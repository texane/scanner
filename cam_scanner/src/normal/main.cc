// use cgal library to estimate point normals

#include <vector>
#include <list>
#include <utility>
#include <fstream>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/IO/write_xyz_points.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef std::pair<Point, Vector> PointVectorPair;

int main(int ac, char** av)
{
  std::ifstream istream(av[1]);
  std::ofstream ostream(av[2]);

  std::list<PointVectorPair> points;

  CGAL::read_xyz_points
  (
   istream,
   std::back_inserter(points),
   CGAL::First_of_pair_property_map<PointVectorPair>()
  );

  // simplification by clustering using erase-remove idiom
  static const double cell_size = 0.001;
  std::list<PointVectorPair>::iterator pos = CGAL::grid_simplify_point_set
  (
   points.begin(), points.end(),
   CGAL::First_of_pair_property_map<PointVectorPair>(),
   cell_size
  );
  points.erase(pos, points.end());

  static const int nb_neighbors = 18; // K-nearest neighbors = 3 rings
  CGAL::pca_estimate_normals
  (
   points.begin(), points.end(),
   CGAL::First_of_pair_property_map<PointVectorPair>(),
   CGAL::Second_of_pair_property_map<PointVectorPair>(),
   nb_neighbors
  );

  std::list<PointVectorPair>::iterator unoriented_points_begin =
    CGAL::mst_orient_normals
    (
     points.begin(), points.end(),
     CGAL::First_of_pair_property_map<PointVectorPair>(),
     CGAL::Second_of_pair_property_map<PointVectorPair>(),
     nb_neighbors
    );

  points.erase(unoriented_points_begin, points.end());

  CGAL::write_xyz_points_and_normals
    (
     ostream,
     points.begin(), points.end(),
     CGAL::First_of_pair_property_map<PointVectorPair>(),
     CGAL::Second_of_pair_property_map<PointVectorPair>()
    );

  return 0;
}
