/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Enhanced OpenSCENARIO parser for Shanghai dataset
 * Extracts multiple vehicle trajectories from .xosc files
 */

#ifndef SHANGHAI_OPENSCENARIO_PARSER_H
#define SHANGHAI_OPENSCENARIO_PARSER_H

#include "ns3/vector.h"
#include "ns3/object.h"
#include <vector>
#include <map>
#include <string>
#include <tinyxml2.h>

namespace ns3 {

struct ScenarioWaypoint
{
  Vector position;
  double time;
  double heading;
  double speed;
};

struct VehicleInfo
{
  std::string vehicleId;
  std::string vehicleType;
  std::string model;
  double length;
  double width;
  double height;
  std::vector<ScenarioWaypoint> waypoints;
  double maxSpeed;
  double averageSpeed;
  double totalDistance;
};

struct ScenarioMetadata
{
  std::string scenarioId;
  std::string name;
  std::string description;
  double duration;
  int totalVehicles;
  std::map<std::string, VehicleInfo> vehicles;
};

class ShanghaiOpenScenarioParser : public Object
{
public:
  static TypeId GetTypeId (void);
  ShanghaiOpenScenarioParser ();
  virtual ~ShanghaiOpenScenarioParser ();

  ScenarioMetadata ParseScenario (const std::string& xoscFile);
  std::vector<std::string> GetAllVehicleIds (const ScenarioMetadata& scenario);
  VehicleInfo GetVehicleTrajectory (const ScenarioMetadata& scenario, const std::string& vehicleId);
  
  // Utility functions for UE selection
  std::vector<std::string> SelectVehiclesByCriteria (const ScenarioMetadata& scenario, 
                                                     int maxVehicles, 
                                                     double minDuration,
                                                     double minSpeed);
  void ExportToNs2Format (const ScenarioMetadata& scenario, 
                          const std::vector<std::string>& selectedVehicles,
                          const std::string& outputPath);
  void ExportToWaypointFormat (const ScenarioMetadata& scenario,
                               const std::vector<std::string>& selectedVehicles, 
                               const std::string& outputPath);

private:
  VehicleInfo ParseSingleVehicle (tinyxml2::XMLElement* scenarioObject, tinyxml2::XMLElement* story);
  ScenarioWaypoint ParseVertex (tinyxml2::XMLElement* vertex, double baseTime);
  std::vector<tinyxml2::XMLElement*> FindAllTrajectoryActions (tinyxml2::XMLElement* story, const std::string& vehicleId);
  double CalculateDistance (const Vector& p1, const Vector& p2) const;
  double CalculateSpeed (const ScenarioWaypoint& wp1, const ScenarioWaypoint& wp2) const;
  void CalculateVehicleStatistics (VehicleInfo& vehicle);
  
  std::string m_currentScenarioFile;
  ScenarioMetadata m_currentScenario;
};

} // namespace ns3

#endif /* SHANGHAI_OPENSCENARIO_PARSER_H */