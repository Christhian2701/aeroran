/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * UE Selection Manager for Shanghai dataset
 * Intelligently selects 63 UEs from available scenarios
 */

#ifndef SHANGHAI_UE_SELECTOR_H
#define SHANGHAI_UE_SELECTOR_H

#include "shanghai-openscenario-parser.h"
#include "ns3/object.h"
#include <vector>
#include <map>
#include <string>

namespace ns3 {

struct UESelectionCriteria
{
  double minTrajectoryDuration;    // Minimum duration in seconds
  double minAverageSpeed;          // Minimum average speed in m/s
  double maxAverageSpeed;          // Maximum average speed in m/s
  double minDistance;              // Minimum total distance in meters
  int maxVehiclesPerScenario;      // Maximum vehicles to take from single scenario
  bool prioritizeDiverseScenarios; // Prefer vehicles from different scenarios
  bool prioritizeActiveVehicles;   // Prefer vehicles with more movement
};

struct SelectedUE
{
  std::string scenarioId;
  std::string vehicleId;
  VehicleInfo vehicleInfo;
  int assignedNodeId;
  double selectionScore;
  std::string selectionReason;
};

class ShanghaiUeSelector : public Object
{
public:
  static TypeId GetTypeId (void);
  ShanghaiUeSelector ();
  virtual ~ShanghaiUeSelector ();

  void SetSelectionCriteria (const UESelectionCriteria& criteria);
  std::vector<SelectedUE> SelectUEs (const std::vector<std::string>& scenarioFiles, int targetUECount);
  
  // Advanced selection methods
  std::vector<SelectedUE> SelectBySpatialDistribution (const std::vector<std::string>& scenarioFiles, 
                                                       int targetUECount,
                                                       double gridSize);
  std::vector<SelectedUE> SelectByTemporalDiversity (const std::vector<std::string>& scenarioFiles, 
                                                     int targetUECount);
  std::vector<SelectedUE> SelectBySpeedCategories (const std::vector<std::string>& scenarioFiles, 
                                                  int targetUECount);
  
  void PrintSelectionSummary (const std::vector<SelectedUE>& selectedUEs) const;
  void ExportSelectionReport (const std::vector<SelectedUE>& selectedUEs, 
                              const std::string& reportPath) const;

private:
  double CalculateSelectionScore (const VehicleInfo& vehicle, const ScenarioMetadata& scenario);
  double CalculateSpatialDiversityScore (const VehicleInfo& vehicle, const std::vector<SelectedUE>& existingUEs);
  double CalculateTemporalDiversityScore (const VehicleInfo& vehicle, const std::vector<SelectedUE>& existingUEs);
  double CalculateSpeedCategoryScore (double speed);
  
  std::vector<ScenarioMetadata> ParseAllScenarios (const std::vector<std::string>& scenarioFiles);
  std::vector<SelectedUE> FilterAndRankVehicles (const std::vector<ScenarioMetadata>& scenarios);
  
  UESelectionCriteria m_criteria;
  std::map<std::string, ScenarioMetadata> m_parsedScenarios;
};

} // namespace ns3

#endif /* SHANGHAI_UE_SELECTOR_H */